import numpy as np
import pandas as pd

from conftest import needs_data
from kestrel_fraud.data import (CLAIM_COLUMNS, TARGET, clean_claims, load_raw, normalise_fault,
                                normalise_note, serial_format)


@needs_data
def test_schema_train_test_reference():
    train, test, partners, products = load_raw()
    assert list(train.columns) == CLAIM_COLUMNS + [TARGET]
    assert list(test.columns) == CLAIM_COLUMNS
    assert {"partner_id", "onboarded_date", "partner_type", "city"} <= set(partners.columns)
    assert {"sku", "list_price_inr", "warranty_months", "family"} <= set(products.columns)
    # every claim joins to reference data
    assert set(train.partner_id) | set(test.partner_id) <= set(partners.partner_id)
    assert set(train.sku) | set(test.sku) <= set(products.sku)


@needs_data
def test_target_values_are_binary_or_undecided():
    train, *_ = load_raw()
    assert set(train[TARGET].dropna().unique()) <= {0.0, 1.0}


@needs_data
def test_clean_labelled_excludes_undecided_and_duplicates(clean):
    labelled, test, all_claims, *_ = clean
    assert labelled[TARGET].isin([0, 1]).all()
    assert labelled["claim_id"].is_unique and test["claim_id"].is_unique
    assert not set(labelled.claim_id) & set(test.claim_id)


def test_resubmission_keeps_first_submission():
    df = pd.DataFrame({
        "claim_id": ["A", "A", "B"], "submitted_at": ["2026-01-05 10:00", "2026-01-02 10:00", "2026-01-03 09:00"],
        "photo_attached": ["Y"] * 3, "partner_inspected": ["N"] * 3, "claim_description": ["blade jammed"] * 3,
        "inspector_note": [np.nan] * 3, "product_serial": ["KH123456789"] * 3, TARGET: [0, 0, 1],
    })
    out = clean_claims(df)
    assert list(out.claim_id) == ["A", "B"]
    assert out.loc[out.claim_id == "A", "ts"].iloc[0] == pd.Timestamp("2026-01-02 10:00")


def test_invalid_dates_raise():
    df = pd.DataFrame({"claim_id": ["A"], "submitted_at": ["not a date"], "photo_attached": ["Y"],
                       "partner_inspected": ["Y"], "claim_description": ["x"], "inspector_note": [None],
                       "product_serial": ["KH1"]})
    try:
        clean_claims(df)
    except ValueError as e:
        assert "submitted_at" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_fault_normalisation_ignores_appended_free_text():
    assert normalise_fault("motor not running. anything at all appended here") == "motor not running"
    assert normalise_fault("Water Leaking") == "water leaking"
    assert normalise_fault("something new") == "other"
    assert normalise_fault(None) == "other"


def test_note_and_serial_normalisation():
    assert normalise_note(np.nan) == "none"
    assert normalise_note("free text the partner typed") == "other"
    assert serial_format("KH123456789") == "canonical"
    assert serial_format("kh123456789") == "lowercase"
    assert serial_format("KH-123456789") == "hyphen"
    assert serial_format(" KH123456789") == "whitespace"
    assert serial_format("12") == "other"
