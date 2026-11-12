import numpy as np

from telco.data import load_churn, make_transactions
from telco.models import choose_threshold


def test_real_churn_data_schema_and_size():
    df = load_churn()
    assert len(df) == 7043
    assert df["customer_id"].is_unique
    assert 0.2 < df["churned"].mean() < 0.3
    assert df["total_charges"].notna().all()


def test_transactions_fraud_rate_and_schema():
    df = make_transactions(n=20000, seed=2)
    assert 0.015 < df["is_fraud"].mean() < 0.025
    assert "is_fraud" in df.columns and "customer_id" not in df.columns


def test_choose_threshold_prefers_catching_fraud_when_fn_costly():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.4, 0.45, 0.9])
    cheap_fn = choose_threshold(y, proba, cost_fn=1.0, cost_fp=5.0)
    costly_fn = choose_threshold(y, proba, cost_fn=50.0, cost_fp=1.0)
    assert costly_fn <= cheap_fn
    assert 0.0 < costly_fn < 1.0


def test_churn_model_beats_chance(services):
    m = services.churn.metrics
    assert m["test_roc_auc"] > 0.8
    assert m["cv_roc_auc_std"] < 0.05


def test_fraud_supervised_beats_unsupervised_baseline(services):
    m = services.fraud.metrics
    assert m["test_pr_auc_supervised"] > m["test_pr_auc_isolation_forest"]
    assert m["test_pr_auc_supervised"] > 3 * m["fraud_rate"]
