import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kestrel_fraud.data import DATA_DIR  # noqa: E402

DATA_FILES = ["train.csv", "test_unlabelled.csv", "partners.csv", "products.csv", "sample_submission.csv"]
HAVE_DATA = all((Path(DATA_DIR) / f).exists() for f in DATA_FILES)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="task-pack CSVs not present (see README: Data)")


@pytest.fixture(scope="session")
def clean():
    if not HAVE_DATA:
        pytest.skip("task-pack CSVs not present")
    from kestrel_fraud.data import load_clean
    return load_clean()


@pytest.fixture(scope="session")
def artifact():
    from kestrel_fraud.config import MODEL_PATH
    if not MODEL_PATH.exists():
        pytest.skip("model artifact missing; run scripts/train_final.py")
    from kestrel_fraud.scoring import load_artifact
    return load_artifact()


@pytest.fixture
def good_claim():
    return dict(submitted_at="2026-09-15 11:00", partner_id="SP3104", sku="KH-MG-02",
                product_serial="KH123456789", days_since_purchase=180, claim_amount_inr=1500.0,
                photo_attached="Y", partner_inspected="N", claim_description="motor not running",
                inspector_note=None, customer_prior_claims=0)
