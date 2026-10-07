"""Final model configuration. Locked on development folds BEFORE the June-2026
holdout was scored (see DECISIONS.md D-12 and VALIDATION.md §3)."""
from pathlib import Path

MODEL_VERSION = "kestrel-fraud-lr-1.0.0"
FINAL_C = 0.3
POST_CHANGE_WEIGHT = 10.0      # weight of post-1-May-2026 rows in training
FINAL_LABEL_CUTOFF = "2026-07-01"   # training export: outcomes for claims up to 30 Jun 2026

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
REPORT_DIR = ROOT / "reports"
