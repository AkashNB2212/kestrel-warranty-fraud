"""One-screen reviewer UI. Calls the API; holds no model itself.

    streamlit run service/app.py
"""
import os
from datetime import datetime

import requests
import streamlit as st

API = os.environ.get("KESTREL_API_URL", "http://127.0.0.1:8000")
SKUS = [f"KH-{f}-0{i}" for f in ["AF", "MG", "WP", "RV", "IC", "CF", "RH"] for i in (1, 2, 3)]
FAULTS = ["motor not running", "blade jammed", "burning smell", "display not working",
          "filter indicator stuck", "loud noise while running", "not charging",
          "power button not working", "remote not working", "tripping mcb", "unit not heating", "water leaking"]
NOTES = ["", "Customer has bill, serial verified", "Heating element open circuit", "Minor fault, part swapped",
         "Motor winding failure confirmed", "PCB replaced under warranty", "Photos match fault, approved",
         "Unit inspected, fault confirmed"]
EXAMPLES = {
    "Blank form": {},
    "Small auto-approved claim, new outlet with fraud history": dict(
        partner_id="SP3160", sku="KH-AF-02", amount=1940.0, inspected="N", photo="Y", note="", prior=0, days=120),
    "Large inspected claim, established outlet": dict(
        partner_id="SP3104", sku="KH-WP-02", amount=4200.0, inspected="Y", photo="Y",
        note="Unit inspected, fault confirmed", prior=0, days=200),
}

st.set_page_config(page_title="Kestrel claim check", page_icon="🔎", layout="centered")
st.title("Warranty claim fraud check")
st.caption("Score one claim before payout. The investigation desk can review 40 claims a month.")

ex = EXAMPLES[st.selectbox("Load an example", list(EXAMPLES))]
with st.form("claim"):
    c1, c2 = st.columns(2)
    partner_id = c1.text_input("Partner ID", ex.get("partner_id", "SP3104"))
    sku = c2.selectbox("SKU", SKUS, index=SKUS.index(ex.get("sku", "KH-MG-02")))
    when = c1.text_input("Submitted at (IST)", datetime(2026, 9, 15, 11, 0).strftime("%Y-%m-%d %H:%M"))
    amount = c2.number_input("Claim amount (Rs)", min_value=1.0, max_value=1_000_000.0,
                             value=float(ex.get("amount", 1500.0)), step=50.0)
    days = c1.number_input("Days since purchase", 0, 3650, int(ex.get("days", 180)))
    prior = c2.number_input("Customer's earlier claims", 0, 100, int(ex.get("prior", 0)))
    inspected = c1.radio("Partner inspected?", ["Y", "N"], index=["Y", "N"].index(ex.get("inspected", "N")), horizontal=True)
    photo = c2.radio("Photo attached?", ["Y", "N"], index=["Y", "N"].index(ex.get("photo", "Y")), horizontal=True)
    fault = c1.selectbox("Fault", FAULTS)
    note = c2.selectbox("Inspector note", NOTES, index=NOTES.index(ex.get("note", "")))
    serial = st.text_input("Product serial", "KH123456789")
    submitted = st.form_submit_button("Score claim", type="primary")

if submitted:
    payload = dict(submitted_at=when, partner_id=partner_id.strip(), sku=sku, product_serial=serial,
                   days_since_purchase=int(days), claim_amount_inr=float(amount), photo_attached=photo,
                   partner_inspected=inspected, claim_description=fault,
                   inspector_note=note or None, customer_prior_claims=int(prior))
    try:
        r = requests.post(f"{API}/predict", json=payload, timeout=10)
    except requests.RequestException:
        st.error(f"Scoring service is not reachable at {API}. Start it with `uvicorn service.api:app --port 8000`.")
        st.stop()
    body = r.json()
    if r.status_code != 200:
        st.error(body.get("message", "Could not score this claim."))
        for d in body.get("details", []):
            st.write(f"- **{d['field']}**: {d['problem']}")
        st.stop()
    colour = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[body["risk_band"]]
    m1, m2 = st.columns(2)
    m1.metric("Fraud risk score", f"{body['fraud_score']:.1%}")
    m2.metric("Risk level", f"{colour} {body['risk_band']}")
    st.info(f"**Suggested action:** {body['suggested_action']}")
    st.subheader("Why")
    for rsn in body["reasons"]:
        st.write(("⬆️ " if rsn["direction"] == "raises risk" else "⬇️ ") + rsn["reason"])
    for w in body.get("warnings", []):
        st.warning(w)
    st.caption(f"Model {body['model']['version']} · investigation outcomes up to {body['model']['trained_on_outcomes_up_to']} · "
               f"HIGH ≥ {body['model']['band_thresholds']['high']:.1%}, MEDIUM ≥ {body['model']['band_thresholds']['medium']:.1%}")
