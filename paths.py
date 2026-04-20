from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

DATASET_DIR = ROOT_DIR / "Dataset"
MISSING_DATASET_DIR = ROOT_DIR / "Dataset_missing"
MODELS_DIR = ROOT_DIR / "models"

ANALYSIS_DIR = ROOT_DIR / "analysis"

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_ANALYSIS_DIR = RESULTS_DIR / "analysis"
RESULTS_BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
RESULTS_EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_LOGS_DIR = RESULTS_DIR / "logs"
RESULTS_REPORTS_DIR = RESULTS_DIR / "reports"
RESULTS_SEED_RUNS_DIR = RESULTS_DIR / "seed_runs"

DOCS_DIR = ROOT_DIR / "docs"
DOCS_ASSETS_DIR = DOCS_DIR / "assets"
DOCS_NOTEBOOKS_DIR = DOCS_DIR / "notebooks"


def ensure_dir(path: Path | str) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
