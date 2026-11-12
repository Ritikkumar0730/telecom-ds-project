"""Churn and fraud models with honest evaluation.

Design choices:
- Gradient-boosted trees (HistGradientBoosting) for tabular data; one-hot for categoricals.
- Metrics that match the business problem: ROC-AUC and PR-AUC for ranking, Brier score for
  calibration, and precision/recall at a fixed alert budget (for fraud review queues).
- Threshold chosen by an explicit cost model (cost of a missed fraud vs. a false alarm),
  not by accuracy.
- Unsupervised IsolationForest kept as a baseline to show the supervised lift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CHURN_CAT = ["contract", "payment_method", "internet_service", "paperless_billing", "partner", "dependents", "tech_support", "online_security"]
CHURN_NUM = ["tenure_months", "monthly_charge", "total_charges", "senior"]

FRAUD_CAT = ["channel"]
FRAUD_NUM = ["amount", "hour", "velocity_1h", "new_device", "distance_km", "account_age_days", "international"]


def _pipeline(cat: list[str], num: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True, random_state=0
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def choose_threshold(y: np.ndarray, proba: np.ndarray, cost_fn: float, cost_fp: float) -> float:
    """Return the probability threshold that minimises expected cost on the given sample."""
    best_t, best_cost = 0.5, np.inf
    for t in np.linspace(0.01, 0.99, 99):
        pred = proba >= t
        fn = np.sum((~pred) & (y == 1))
        fp = np.sum(pred & (y == 0))
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    features: list[str]
    threshold: float
    metrics: dict = field(default_factory=dict)
    importances: dict = field(default_factory=dict)

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(rows[self.features])[:, 1]


def train_churn(df: pd.DataFrame, seed: int = 0) -> TrainedModel:
    features = CHURN_CAT + CHURN_NUM
    X, y = df[features], df["churned"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    pipe = _pipeline(CHURN_CAT, CHURN_NUM)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    perm = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=seed)
    importances = dict(sorted(zip(features, perm.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))

    threshold = choose_threshold(y_te, proba, cost_fn=5.0, cost_fp=1.0)
    return TrainedModel(
        name="churn",
        pipeline=pipe,
        features=features,
        threshold=threshold,
        metrics={
            "base_rate": round(float(y.mean()), 4),
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
            "test_pr_auc": round(float(average_precision_score(y_te, proba)), 4),
            "test_brier": round(float(brier_score_loss(y_te, proba)), 4),
            "threshold": round(threshold, 2),
        },
        importances=importances,
    )


def train_fraud(df: pd.DataFrame, seed: int = 0, alert_budget: float = 0.01) -> TrainedModel:
    features = FRAUD_CAT + FRAUD_NUM
    X, y = df[features], df["is_fraud"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    pipe = _pipeline(FRAUD_CAT, FRAUD_NUM)
    pipe.set_params(clf__class_weight="balanced")
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    # Unsupervised baseline: anomaly score from IsolationForest on the same numeric features.
    iso_pre = pipe.named_steps["pre"]
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(iso_pre.transform(X_tr)[:, :].astype(float))
    iso_score = -iso.score_samples(iso_pre.transform(X_te).astype(float))

    # Operating point for a review team that can only look at `alert_budget` of transactions.
    k = max(1, int(alert_budget * len(proba)))
    top = np.argsort(-proba)[:k]
    caught = int(y_te[top].sum())
    threshold = choose_threshold(y_te, proba, cost_fn=50.0, cost_fp=1.0)

    perm = permutation_importance(pipe, X_te, y_te, scoring="average_precision", n_repeats=3, random_state=seed)
    importances = dict(sorted(zip(features, perm.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))

    return TrainedModel(
        name="fraud",
        pipeline=pipe,
        features=features,
        threshold=threshold,
        metrics={
            "fraud_rate": round(float(y.mean()), 4),
            "test_pr_auc_supervised": round(float(average_precision_score(y_te, proba)), 4),
            "test_pr_auc_isolation_forest": round(float(average_precision_score(y_te, iso_score)), 4),
            "test_roc_auc_supervised": round(float(roc_auc_score(y_te, proba)), 4),
            f"recall_at_{int(alert_budget * 100)}pct_alerts": round(caught / max(1, y_te.sum()), 4),
            f"precision_at_{int(alert_budget * 100)}pct_alerts": round(caught / k, 4),
            "threshold": round(threshold, 2),
        },
        importances=importances,
    )
