"""Fold-local preprocessing and fixed, deliberately small model families."""

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .data import PROTOCOL
from .features import ENDPOINT, SPREAD
from .splits import inner_folds


class Representation:
    def __init__(self, regime=False, interactions=False):
        self.regime = regime
        self.interactions = interactions

    def fit(self, frame):
        self.cols = list(frame.columns)
        self.macro = [c for c in self.cols if c not in SPREAD + ENDPOINT]
        if self.regime:
            self.mscale = StandardScaler().fit(frame[self.macro])
            z = self.mscale.transform(frame[self.macro])
            self.cluster = KMeans(
                n_clusters=2, n_init=PROTOCOL["kmeans_n_init"], random_state=PROTOCOL["seed"]
            ).fit(z)
            order = sorted(range(2), key=lambda k: tuple(self.cluster.cluster_centers_[k]))
            self.mapping = np.argsort(order)
            self.occupancy = np.bincount(self.mapping[self.cluster.labels_], minlength=2).tolist()
        self.scale = StandardScaler().fit(self.expand(frame))
        return self

    def expand(self, frame):
        z = frame[self.cols].to_numpy()
        if self.regime:
            r = self.mapping[self.cluster.predict(self.mscale.transform(frame[self.macro]))]
            z = np.column_stack([z, r])
            if self.interactions:
                z = np.column_stack([z, frame[self.macro].to_numpy() * r[:, None]])
        return z

    def transform(self, frame):
        return self.scale.transform(self.expand(frame))


def choose_alpha(frame, y, outcomes, regime):
    losses = {a: [] for a in PROTOCOL["ridge_alphas"]}
    for tr, va in inner_folds(frame.index, outcomes, PROTOCOL["minimum_inner_train"]):
        representation = Representation(regime, regime).fit(frame.loc[tr])
        x, v = representation.transform(frame.loc[tr]), representation.transform(frame.loc[va])
        for alpha in losses:
            model = Ridge(alpha=alpha).fit(x, y.loc[tr])
            losses[alpha].extend((model.predict(v) - y.loc[va].to_numpy()) ** 2)
    valid = {a: np.mean(v) for a, v in losses.items() if v}
    if not valid:
        raise ValueError("Insufficient chronological inner validation")
    return min(valid, key=lambda a: (valid[a], -a))


def fitted_model(frame, y, outcomes, family, regime, task, seed=42):
    classification = task != "level"
    if classification and y.nunique() < 2:
        return (
            None,
            None,
            {"fallback": "single_class_prevalence", "constant": (y.sum() + 1) / (len(y) + 2)},
        )
    rep = Representation(regime, regime and family != "rf").fit(frame)
    x = rep.transform(frame)
    info = {"fallback": "", "occupancy": getattr(rep, "occupancy", None)}
    if family == "ridge":
        alpha = choose_alpha(frame, y, outcomes, regime)
        model = Ridge(alpha=alpha)
        info["alpha"] = alpha
    elif family == "logistic":
        model = LogisticRegression(
            C=PROTOCOL["logistic_c"], max_iter=PROTOCOL["logistic_max_iter"], random_state=seed
        )
    else:
        cls = RandomForestClassifier if classification else RandomForestRegressor
        model = cls(
            n_estimators=PROTOCOL["rf_trees"],
            min_samples_leaf=PROTOCOL["rf_minimum_leaf"],
            max_features=1.0,
            random_state=seed,
            n_jobs=1,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    return rep, model, info


def predict(fit, frame, task):
    rep, model, info = fit
    if model is None:
        return np.full(len(frame), info["constant"])
    x = rep.transform(frame)
    return model.predict(x) if task == "level" else model.predict_proba(x)[:, 1]


def baseline(name, target, outcomes, train, origin, features, h, q, task, cutoff=None):
    if name == "mean_persistence":
        return float(features.loc[origin, "mean"])
    if name in {"endpoint_persistence", "raw_spread"}:
        return float(features.loc[origin, "endpoint"])
    if name == "training_mean":
        return float(outcomes.loc[train, "actual"].mean())
    if name == "ar1":
        x = features.loc[train, "mean"].to_numpy()
        y = outcomes.loc[train, "actual"].to_numpy()
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        return float(coef @ [1, features.loc[origin, "mean"]])
    if name == "prevalence":
        y = outcomes.loc[train, "actual"]
        return float((y.sum() + 1) / (len(y) + 2))
    if name != "transition":
        raise ValueError(f"Unknown baseline: {name}")
    history = target.loc[: origin if cutoff is None else cutoff]
    state = history.ge(q).astype(int)
    counts = np.ones((2, 2))
    for i in range(1, len(history)):
        if history.iloc[i - 1 : i + 1].notna().all():
            counts[state.iloc[i - 1], state.iloc[i]] += 1
    matrix = counts / counts.sum(axis=1, keepdims=True)
    return float(
        1 - matrix[0, 0] ** h
        if task == "entry"
        else np.linalg.matrix_power(matrix, h)[int(target.loc[origin] >= q), 1]
    )
