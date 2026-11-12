"""Builds all trained artifacts once and holds them for the API and the agent."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from telco.data import load_churn, make_transactions
from telco.models import TrainedModel, train_churn, train_fraud
from telco.rag import Retriever, evaluate_retrieval, load_chunks, load_eval_questions


@dataclass
class Services:
    customers: pd.DataFrame
    churn: TrainedModel
    fraud: TrainedModel
    retriever: Retriever
    rag_metrics: dict

    @classmethod
    def build(cls, seed: int = 42) -> "Services":
        customers = load_churn()
        transactions = make_transactions(seed=seed + 1)
        churn = train_churn(customers, seed=0)
        fraud = train_fraud(transactions, seed=0)
        retriever = Retriever(load_chunks())
        rag_metrics = evaluate_retrieval(retriever, load_eval_questions(), k=3)
        return cls(customers=customers, churn=churn, fraud=fraud, retriever=retriever, rag_metrics=rag_metrics)

    def churn_risk(self, customer_id: str) -> dict:
        row = self.customers.loc[self.customers["customer_id"] == customer_id]
        if row.empty:
            return {"error": f"customer {customer_id} not found"}
        p = float(self.churn.predict_proba(row)[0])
        r = row.iloc[0]
        return {
            "customer_id": customer_id,
            "churn_probability": round(p, 3),
            "high_risk": p >= self.churn.threshold,
            "contract": r["contract"],
            "tenure_months": int(r["tenure_months"]),
            "monthly_charge": float(r["monthly_charge"]),
            "internet_service": r["internet_service"],
            "tech_support": r["tech_support"],
        }

    def fraud_score(self, transaction: dict) -> dict:
        row = pd.DataFrame([transaction])
        p = float(self.fraud.predict_proba(row)[0])
        return {
            "fraud_probability": round(p, 4),
            "flagged": p >= self.fraud.threshold,
            "threshold": self.fraud.threshold,
        }

    def search_policy(self, query: str, k: int = 3) -> dict:
        results = self.retriever.search(query, k=k)
        return {
            "results": [
                {"doc_id": c.doc_id, "section": c.section, "score": round(s, 3), "text": c.text}
                for c, s in results
            ]
        }
