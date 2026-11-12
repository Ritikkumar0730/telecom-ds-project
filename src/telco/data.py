"""Data loaders.

Churn: real public data, IBM Telco Customer Churn (7,043 customers, one row per customer,
label = churned in the most recent month). Source: https://github.com/IBM/telco-customer-churn-on-icp4d
Fraud: synthetic transactions (no public fraud dataset is bundled yet). The class-conditional
generative process is documented below so its limits are explicit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IBM_CHURN_PATH = ROOT / "data" / "raw" / "telco_churn_ibm.csv"
CHANNELS = ["app", "web", "pos"]


def load_churn(path: Path = IBM_CHURN_PATH) -> pd.DataFrame:
    """Load IBM Telco churn data into the feature schema used by the models."""
    raw = pd.read_csv(path)
    total = pd.to_numeric(raw["TotalCharges"].astype(str).str.strip(), errors="coerce")
    return pd.DataFrame(
        {
            "customer_id": raw["customerID"],
            "contract": raw["Contract"],
            "payment_method": raw["PaymentMethod"],
            "internet_service": raw["InternetService"],
            "paperless_billing": raw["PaperlessBilling"],
            "partner": raw["Partner"],
            "dependents": raw["Dependents"],
            "tech_support": raw["TechSupport"],
            "online_security": raw["OnlineSecurity"],
            "tenure_months": raw["tenure"].astype(int),
            "monthly_charge": raw["MonthlyCharges"].astype(float),
            "total_charges": total.fillna(0.0).astype(float),
            "senior": raw["SeniorCitizen"].astype(int),
            "churned": (raw["Churn"] == "Yes").astype(int),
        }
    )


def make_transactions(n: int = 50_000, fraud_rate: float = 0.02, seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC payment transactions with a rare fraud label.

    Fraud rows are drawn from overlapping class-conditional distributions. Overlap is deliberate:
    most fraud looks ordinary on any single feature, so models must combine weak signals.
    Without overlap the task is trivial. Metrics on this data show pipeline correctness only.
    """
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate

    def pick(legit, fraud):
        return np.where(is_fraud, fraud, legit)

    amount = np.where(is_fraud, rng.lognormal(5.4, 1.0, size=n), rng.lognormal(4.2, 0.9, size=n))
    night = rng.random(n) < pick(0.12, 0.35)
    hour = np.where(night, rng.choice([0, 1, 2, 3, 4, 23], size=n), rng.integers(0, 24, size=n))
    velocity_1h = np.where(is_fraud, rng.poisson(2.2, size=n), rng.poisson(0.9, size=n))
    new_device = rng.random(n) < pick(0.08, 0.30)
    distance_km = np.where(is_fraud, rng.exponential(35.0, size=n), rng.exponential(8.0, size=n))
    account_age_days = np.where(is_fraud, rng.integers(1, 500, size=n), rng.integers(30, 3000, size=n))
    international = rng.random(n) < pick(0.03, 0.18)
    channel = rng.choice(CHANNELS, size=n, p=[0.5, 0.2, 0.3])
    channel = np.where(is_fraud & (rng.random(n) < 0.5), "web", channel)

    return pd.DataFrame(
        {
            "transaction_id": [f"T{i:07d}" for i in range(1, n + 1)],
            "amount": amount.round(2),
            "hour": hour,
            "velocity_1h": velocity_1h,
            "new_device": new_device.astype(int),
            "distance_km": distance_km.round(1),
            "account_age_days": account_age_days,
            "international": international.astype(int),
            "channel": channel,
            "is_fraud": is_fraud.astype(int),
        }
    )
