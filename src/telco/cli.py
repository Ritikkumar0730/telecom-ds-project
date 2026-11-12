"""Command line: train all models, evaluate RAG retrieval, and write a metrics report."""

from __future__ import annotations

import json
from pathlib import Path

from telco.services import Services

REPORT = Path(__file__).resolve().parents[2] / "reports" / "metrics.json"


def main() -> None:
    services = Services.build()
    report = {
        "churn": services.churn.metrics,
        "churn_top_features": dict(list(services.churn.importances.items())[:5]),
        "fraud": services.fraud.metrics,
        "fraud_top_features": dict(list(services.fraud.importances.items())[:5]),
        "rag": {k: v for k, v in services.rag_metrics.items() if k != "misses"},
        "rag_misses": services.rag_metrics["misses"],
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
