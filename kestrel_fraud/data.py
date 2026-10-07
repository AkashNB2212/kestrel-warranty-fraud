"""Loading and cleaning of the Kestrel warranty-claims pack.

Every rule here is traceable to the pack (README.txt, email-thread.txt, ops-policy.pdf);
see DECISIONS.md for the reasoning behind each one.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("KESTREL_DATA_DIR", Path(__file__).resolve().parent.parent))

# Policy §5: from 1 May 2026 claims under Rs 2,000 are auto-approved without inspection.
AUTO_APPROVAL_START = pd.Timestamp("2026-05-01")
AUTO_APPROVAL_LIMIT_INR = 2000.0
# Policy §9: Zoho -> CRM cut-over.
CRM_START = pd.Timestamp("2025-10-01")

CLAIM_COLUMNS = [
    "claim_id", "submitted_at", "partner_id", "sku", "product_serial",
    "days_since_purchase", "claim_amount_inr", "photo_attached", "partner_inspected",
    "claim_description", "inspector_note", "customer_prior_claims", "source",
]
TARGET = "is_fraud"

# The fault vocabulary is a closed list of phrases typed via a dropdown-like field.
# Anything appended after the phrase (free text) is ignored on purpose: it is
# unverified partner input and must never steer the model.
FAULT_PHRASES = [
    "blade jammed", "burning smell", "display not working", "display blank",
    "filter indicator stuck", "loud noise while running", "motor not running",
    "not charging", "power button not working", "remote not working",
    "tripping mcb", "unit not heating", "water leaking",
]
INSPECTOR_NOTES = [
    "Customer has bill, serial verified", "Heating element open circuit",
    "Minor fault, part swapped", "Motor winding failure confirmed",
    "PCB replaced under warranty", "Photos match fault, approved",
    "Unit inspected, fault confirmed",
]


def load_raw(data_dir: Path | str = DATA_DIR):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_unlabelled.csv")
    partners = pd.read_csv(data_dir / "partners.csv")
    products = pd.read_csv(data_dir / "products.csv")
    return train, test, partners, products


def normalise_fault(text) -> str:
    if not isinstance(text, str):
        return "other"
    t = text.strip().lower()
    # Longest phrase first so "display not working" wins over "display blank" etc.
    for phrase in sorted(FAULT_PHRASES, key=len, reverse=True):
        if t.startswith(phrase):
            return "display not working" if phrase == "display blank" else phrase
    return "other"


def normalise_note(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return "none"
    t = text.strip()
    return t if t in INSPECTOR_NOTES else "other"


def serial_format(s) -> str:
    if not isinstance(s, str):
        return "missing"
    if re.fullmatch(r"KH\d{9}", s):
        return "canonical"
    if re.fullmatch(r"kh\d{9}", s):
        return "lowercase"
    if re.fullmatch(r"KH-\d{9}", s):
        return "hyphen"
    if re.fullmatch(r"\s*KH\d{9}\s*", s, flags=re.IGNORECASE):
        return "whitespace"
    return "other"


def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast, normalise and de-duplicate claims (works on train or test rows)."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["submitted_at"], errors="coerce")
    if df["ts"].isna().any():
        raise ValueError(f"{int(df['ts'].isna().sum())} rows have an unparseable submitted_at")
    # Email (Tanmay): bounced claims are re-submitted with the same claim number.
    # The fraud decision is taken at the first submission, so keep that one.
    df = df.sort_values(["ts", "claim_id"]).drop_duplicates("claim_id", keep="first")
    df["photo"] = (df["photo_attached"].astype(str).str.upper() == "Y").astype(int)
    df["inspected"] = (df["partner_inspected"].astype(str).str.upper() == "Y").astype(int)
    df["fault"] = df["claim_description"].map(normalise_fault)
    df["note"] = df["inspector_note"].map(normalise_note)
    df["serial_fmt"] = df["product_serial"].map(serial_format)
    if TARGET in df.columns:
        # Email (Tanmay): blank = still under investigation. Excluded from training/eval.
        df["label_status"] = np.where(df[TARGET].isna(), "undecided", "decided")
    return df.reset_index(drop=True)


def attach_reference(df: pd.DataFrame, partners: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    p = partners.copy()
    p["onboarded"] = pd.to_datetime(p["onboarded_date"], errors="coerce")
    out = df.merge(p[["partner_id", "city", "partner_type", "onboarded"]], on="partner_id", how="left")
    out = out.merge(products, on="sku", how="left")
    return out


def load_clean(data_dir: Path | str = DATA_DIR):
    """Return (labelled_train, test, all_claims, partners, products).

    all_claims = train (incl. undecided) + test, de-duplicated; used only for
    label-free partner activity features, which are known at submission time.
    """
    train, test, partners, products = load_raw(data_dir)
    tr = attach_reference(clean_claims(train), partners, products)
    te = attach_reference(clean_claims(test), partners, products)
    tr["split"] = "train"
    te["split"] = "test"
    all_claims = pd.concat([tr, te], ignore_index=True, sort=False)
    labelled = tr[tr["label_status"] == "decided"].copy()
    labelled[TARGET] = labelled[TARGET].astype(int)
    return labelled, te, all_claims, partners, products
