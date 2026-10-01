"""Execute the six study readers in isolated kernels using the current interpreter."""

import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def execute():
    """Save executed readers and their validation record; leave legacy notebooks intact."""
    kernels = Path(sys.prefix) / "share/jupyter/kernels/research-study"
    kernels.mkdir(parents=True, exist_ok=True)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Research study (project venv)",
        "language": "python",
    }
    (kernels / "kernel.json").write_text(json.dumps(spec))
    os.environ["JUPYTER_PATH"] = str(Path(sys.prefix) / "share/jupyter")
    records = []
    for path in sorted((ROOT / "notebooks/v2").glob("0[1-6]_*.ipynb")):
        start = time.monotonic()
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=120,
            kernel_name="research-study",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        assert all(
            cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code"
        )
        nbformat.write(notebook, path)
        records.append({"path": str(path.relative_to(ROOT)), "seconds": time.monotonic() - start})
        print(f"EXECUTED {path.name}", flush=True)
    report = ROOT / "research/study_results/validation.json"
    validation = json.loads(report.read_text())
    validation["notebooks"] = {"status": "passed", "count": len(records), "records": records}
    report.write_text(json.dumps(validation, indent=2) + "\n")


if __name__ == "__main__":
    execute()
