"""Every experiment writes its results to its own new folder, results/<experiment>/.

Results are never overwritten or deleted, so the report can show every variant that was tried,
including the ones that did not work. Each experiment is also logged in RESULTS.md.
"""
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def experiment_dir(name):
    """Folder for a new experiment's results; stops if it already exists.

    The folder is not created here: create it once the results are ready (out.mkdir(parents=True)),
    so a run that crashes leaves nothing behind.
    """
    path = RESULTS_DIR / name
    if path.exists():
        raise SystemExit(f"{path} already exists. Results are never overwritten: choose a new experiment name.")
    return path
