"""Small forecast jobs with immutable manifests and atomic resumable checkpoints."""

import hashlib
import json
import platform
import shutil
import subprocess
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import PROTOCOL, ROOT, load_data
from .features import BLOCKS, column_names, labels
from .models import baseline, fitted_model, predict
from .splits import training_index


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    tmp.replace(path)


def code_hash():
    paths = sorted(
        p for p in Path(__file__).parent.iterdir() if p.suffix in {".py", ".json", ".lock"}
    )
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in paths)).hexdigest()


def environment():
    import importlib.metadata

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
        },
    }


def scenarios(profile):
    base = {
        "name": "core",
        "target": "mean_observed",
        "delay": False,
        "quantile": 0.75,
        "five_macro": False,
        "extension": False,
        "refit": 1,
        "rolling": None,
        "seed": 42,
    }
    if profile in {"core", "smoke"}:
        return [base]
    if profile == "external":
        return [{**base, "name": "external", "five_macro": True, "extension": True}]
    changes = [
        {"name": "annual", "refit": 12},
        {"name": "rolling120", "rolling": 120},
        {"name": "legacy", "target": "mean_legacy_ffill"},
        {"name": "delay", "delay": True},
        {"name": "q70", "quantile": 0.7},
        {"name": "q80", "quantile": 0.8},
        {"name": "five_macro", "five_macro": True},
    ]
    changes += [{"name": f"seed{s}", "seed": s} for s in range(5)]
    return [{**base, **change} for change in changes]


def job_specs(profile):
    specs = []
    for scenario in scenarios(profile):
        name = scenario["name"]
        tasks = ["state", "entry"] if name in {"q70", "q80"} else ["level"]
        if profile in {"core", "smoke"}:
            tasks = ["level", "state", "entry"]
        for h in PROTOCOL["horizons"]:
            for task in tasks:
                families = ["ridge", "rf"] if task == "level" else ["logistic", "rf"]
                blocks = BLOCKS if profile in {"core", "smoke"} or task != "level" else BLOCKS[2:]
                if name.startswith("seed"):
                    families = ["rf"]
                combinations = [(family, block) for family in families for block in blocks]
                if task == "level":
                    combinations += [
                        (b, "S_last")
                        for b in [
                            "mean_persistence",
                            "endpoint_persistence",
                            "training_mean",
                            "ar1",
                        ]
                    ]
                else:
                    combinations += [
                        (b, "S_last")
                        for b in ["prevalence", "transition", "single_logistic", "raw_spread"]
                    ]
                for family, block in combinations:
                    job = {
                        **scenario,
                        "h": h,
                        "task": task,
                        "model": family,
                        "block": block,
                        "smoke": profile == "smoke",
                    }
                    job["id"] = hashlib.sha256(
                        json.dumps(job, sort_keys=True).encode()
                    ).hexdigest()[:16]
                    specs.append(job)
    return specs


def data_for(job):
    return load_data(
        **{k: job[k] for k in ["target", "delay", "quantile", "five_macro", "extension"]}
    )


def origin_record(job, data, outcomes, origin):
    actual = outcomes.loc[origin, "actual"]
    return {
        "origin": origin,
        "scenario": job["name"],
        "target_definition": job["target"],
        "horizon": job["h"],
        "task": job["task"],
        "model": job["model"],
        "feature_block": job["block"],
        "seed": job["seed"],
        "threshold_value": data.threshold,
        "threshold_training_end": PROTOCOL["threshold_end"],
        "actual": actual,
        "target_date": outcomes.loc[origin, "target_date"],
        "label_available": outcomes.loc[origin, "label_available"],
        "target_available_at": outcomes.loc[origin, "label_available"],
        "at_risk": bool(
            outcomes.loc[origin, "risk"]
            and data.current_known.loc[origin]
            and outcomes.loc[origin, "path_observed"]
        ),
        "risk_set": "at_risk" if job["task"] == "entry" else "all",
        "score_type": (
            "ranking"
            if job["model"] == "raw_spread"
            else ("level_pp" if job["task"] == "level" else "probability")
        ),
        "prediction": np.nan,
        "eligible": False,
        "exclusion_reason": "",
        "fallback": "",
    }


def exclusion(data, outcomes, origin, job):
    if data.features.loc[origin].isna().any():
        return "unavailable_features"
    if pd.isna(outcomes.loc[origin, "actual"]):
        return "unknown_future"
    if job["task"] == "entry" and not (
        outcomes.loc[origin, "risk"] and data.current_known.loc[origin]
    ):
        return "outside_risk_set"
    return ""


def forecast_job(job):
    with threadpool_limits(limits=1):
        return forecast_job_inner(job)


def forecast_job_inner(job):
    data = data_for(job)
    outcomes = labels(data.target, data.available, job["h"], job["task"], data.threshold)
    cols = column_names(data.features, job["block"])
    if job["model"] == "single_logistic":
        cols = ["endpoint"]
    frame = data.features[cols]
    start = (
        PROTOCOL["evaluation_start"] if job["task"] == "level" else PROTOCOL["calibration_start"]
    )
    origins = data.target.loc[start:].index
    if job["extension"]:
        origins = origins[origins > pd.Timestamp(PROTOCOL["end"])]
    if job["smoke"]:
        origins = origins[origins <= pd.Timestamp("2009-12-31")]
    records, cache = [], {}
    for origin in origins:
        record = origin_record(job, data, outcomes, origin)
        reason = exclusion(data, outcomes, origin, job)
        if reason:
            record["exclusion_reason"] = reason
        else:
            try:
                record.update(forecast_origin(job, data, outcomes, frame, origin, cache))
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                record["exclusion_reason"] = f"model_failure: {type(exc).__name__}: {exc}"
        records.append(record)
    return pd.DataFrame(records)


def forecast_origin(job, data, outcomes, frame, origin, cache):
    cutoff = origin
    if job["refit"] == 12:
        cutoff = pd.Timestamp(origin.year, 1, 31)
    if job["extension"]:
        cutoff = pd.Timestamp(PROTOCOL["end"])
    tr = training_index(data.features, outcomes, cutoff, job["rolling"], job["task"] == "entry")
    minimum = (
        PROTOCOL["minimum_inner_train"]
        if origin < pd.Timestamp(PROTOCOL["evaluation_start"])
        else PROTOCOL["minimum_train"]
    )
    # Entry training has fewer rows by design; require 60 eligible at-risk examples.
    if job["task"] == "entry":
        minimum = 60
    if len(tr) < minimum:
        return {"exclusion_reason": "insufficient_training"}
    info = {}
    if job["model"] in {"ridge", "rf", "logistic", "single_logistic"}:
        value, info = learned_prediction(job, data, outcomes, frame, origin, cutoff, tr, cache)
    else:
        value = baseline(
            job["model"],
            data.target,
            outcomes,
            tr,
            origin,
            data.features,
            job["h"],
            data.threshold,
            job["task"],
            cutoff=cutoff,
        )
    if not np.isfinite(value):
        raise ValueError("Nonfinite prediction")
    return {
        "prediction": value,
        "eligible": True,
        "cutoff_timestamp": cutoff,
        "train_start": tr.min(),
        "train_end": tr.max(),
        "last_training_label_available_at": outcomes.loc[tr, "label_available"].max(),
        "tuning_end": cutoff,
        "regime_fit_end": cutoff if job["block"].endswith("+R") else None,
        "fallback": info.get("fallback", ""),
        "fit_details": json.dumps(info),
    }


def learned_prediction(job, data, outcomes, frame, origin, cutoff, train, cache):
    """Fit once per refit cutoff; cache only that fit to bound forest memory."""
    anchor = "mean" if job["block"] == "S_mean" else "endpoint"
    if cutoff not in cache:
        cache.clear()
        y = outcomes.loc[train, "actual"].copy()
        if job["task"] == "level":
            y -= data.features.loc[train, anchor]
        family = "logistic" if job["model"] == "single_logistic" else job["model"]
        cache[cutoff] = fitted_model(
            frame.loc[train],
            y,
            outcomes,
            family,
            job["block"].endswith("+R"),
            job["task"],
            job["seed"],
        )
    fit = cache[cutoff]
    value = float(predict(fit, frame.loc[[origin]], job["task"])[0])
    if job["task"] == "level":
        value += float(data.features.loc[origin, anchor])
    return value, fit[2]


def manifest(profile):
    study = load_data(extension=profile == "external", five_macro=profile == "external")
    return {
        "profile": profile,
        "code_hash": code_hash(),
        "protocol": PROTOCOL,
        "input_hashes": study.hashes,
        "environment": environment(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "quality": study.quality,
        "jobs": job_specs(profile),
    }


def execute(profile="smoke", resume=None, workers=4, hours=4):
    if not 1 <= workers <= 4 or not 0 < hours <= 4:
        raise ValueError("Use 1–4 workers and a positive batch length <=4 hours")
    current = manifest(profile)
    if resume:
        out = ROOT / "results/research" / resume
        old = json.loads((out / "manifest.json").read_text())
        if any(
            old[k] != current[k]
            for k in ["profile", "code_hash", "protocol", "input_hashes", "environment"]
        ):
            raise ValueError("Resume manifest mismatch; start a new run")
        status_path = out / "status.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("complete"):
            print(f"RESULT {out.name}: already complete; no files changed", flush=True)
            return out
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        out = ROOT / "results/research" / run_id
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(out / "manifest.json", current)
        (out / "jobs").mkdir()
        for path in Path(__file__).parent.iterdir():
            if path.is_file():
                destination = out / "source/research/study" / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        (out / "working_tree.patch").write_bytes(
            subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
        )
        data = load_data(extension=profile == "external", five_macro=profile == "external")
        data.availability_ledger.to_csv(out / "availability.csv", index=False)
    pending = [j for j in current["jobs"] if not (out / "jobs" / (j["id"] + ".csv")).exists()]
    start = time.monotonic()
    print(f"RUN {out.name}: {len(pending)} jobs pending", flush=True)
    run_jobs(pending, out, workers, start + hours * 3600)
    files = sorted((out / "jobs").glob("*.csv"))
    if files:
        predictions = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        predictions["run_id"] = out.name
        predictions["config_hash"] = hashlib.sha256(
            json.dumps(PROTOCOL, sort_keys=True).encode()
        ).hexdigest()
        predictions["input_hash"] = hashlib.sha256(
            json.dumps(current["input_hashes"], sort_keys=True).encode()
        ).hexdigest()
        keys = ["origin", "scenario", "horizon", "task", "model", "feature_block", "seed"]
        if predictions.duplicated(keys).any():
            raise ValueError("Duplicate forecast keys")
        tmp = out / "predictions.tmp"
        predictions.to_csv(tmp, index=False)
        tmp.replace(out / "predictions.csv")
    complete = len(files) == len(current["jobs"])
    atomic_json(
        out / "status.json",
        {
            "complete": complete,
            "finished_jobs": len(files),
            "total_jobs": len(current["jobs"]),
            "batch_seconds": time.monotonic() - start,
        },
    )
    print(f"RESULT {out.name}: complete={complete}", flush=True)
    return out


def run_jobs(pending, out, workers, deadline):
    iterator = iter(pending)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active = {}
        while time.monotonic() < deadline or active:
            while len(active) < workers and time.monotonic() < deadline:
                job = next(iterator, None)
                if job is None:
                    break
                active[pool.submit(forecast_job, job)] = job
            if not active:
                break
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                frame = future.result()
                tmp = out / "jobs" / (job["id"] + ".tmp")
                frame.to_csv(tmp, index=False)
                tmp.replace(tmp.with_suffix(".csv"))
                print(
                    f"DONE {job['name']} {job['task']} h{job['h']} {job['model']} {job['block']}",
                    flush=True,
                )
