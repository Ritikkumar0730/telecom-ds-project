# Telecom Data Science: Churn, Fraud, RAG and Agentic AI

An end-to-end project covering the stack a telecom data science team uses: predictive models for customer churn and payment fraud, a retrieval-augmented (RAG) policy assistant, and a tool-using LLM agent that combines them behind a REST API.

**Data:**
- **Churn:** real public data, [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (7,043 customers, stored at `data/raw/telco_churn_ibm.csv`). Check the source repo's license before reusing the file.
- **Fraud:** **synthetic** transactions. No public fraud dataset is bundled yet, so fraud metrics show pipeline correctness, not real-world performance.
- **Policy documents:** **illustrative**, written for this project, not any operator's real terms.

## What it covers

| Area | Implementation |
|---|---|
| Churn prediction | Gradient-boosted trees on real data, 5-fold CV, permutation importance, cost-based threshold |
| Fraud detection | Class-weighted gradient boosting vs. IsolationForest baseline, alert-budget precision/recall, cost-based threshold |
| Statistics and evaluation | ROC-AUC, PR-AUC, Brier score, CV variance, permutation importance |
| RAG | Section-aware chunking; TF-IDF and LSA (latent semantic) retrieval; recall@k and MRR on a labelled question set |
| Agentic AI | Claude tool use with three tools (churn lookup, fraud scoring, policy search) and a step cap; offline router fallback |
| Agent evaluation | Labelled cases scoring tool-selection accuracy and answer content |
| Serving | FastAPI with typed request validation; health endpoint reports model metrics; Dockerfile |
| Engineering | Package layout, seeded pipelines, pytest suite (16 tests), GitHub Actions CI |

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m telco.cli              # train, evaluate, write reports/metrics.json
PYTHONPATH=src python -m telco.eval_agent       # agent tool-selection evaluation
PYTHONPATH=src python -m pytest -q              # run tests
PYTHONPATH=src uvicorn telco.api:app --reload   # serve the API on :8000
```

Or with Docker:

```bash
docker build -t telco-ds . && docker run -p 8000:8000 telco-ds
```

Set `ANTHROPIC_API_KEY` to run the agent with Claude (model defaults to `claude-sonnet-5-5`; override with `TELCO_LLM_MODEL`). Without it, the agent uses an offline router that calls the same tools, so the project runs with no network or cost.

## API

- `GET /health`: status, agent mode, churn/fraud/RAG metrics
- `GET /customers/{customer_id}/churn-risk`: e.g. `7590-VHVEG`
- `POST /transactions/fraud-score`
- `POST /policy/search`
- `POST /ask`: natural-language question answered by the agent, with a tool-call trace

## Results (from `reports/metrics.json`)

- **Churn (real IBM data, 26.5% base rate):** test ROC-AUC 0.841, PR-AUC 0.638; 5-fold CV ROC-AUC 0.840 ± 0.013. Top drivers: contract type, tenure, monthly charge.
- **Fraud (synthetic, 2.1% base rate):** supervised PR-AUC 0.826 vs. IsolationForest 0.523. At a 1% alert budget: precision 0.95, recall 0.45.
- **Policy RAG (26 questions):** recall@3 0.96, MRR 0.94 for both TF-IDF and LSA. LSA gave no gain on this set.
- **Agent (7 labelled cases, offline router):** tool-selection accuracy 1.00. This is a small set; treat it as a smoke test.

## Limitations

- **Fraud data is synthetic.** Its class overlap was tuned so the task isn't trivial, but real fraud data is harder and noisier. Replace it with a real dataset before drawing conclusions.
- **RAG questions were written by the author, partly alongside the documents.** The score is optimistic. An independent question set would be a fairer test.
- **Retrieval is lexical or LSA.** Neither matched the other on this set. A dense neural embedding model would be the next comparison.
- **The agent's Claude mode is covered by a stubbed-client test only.** Run it with a real key and check traces before claiming it works.
- **No experiment tracking or model registry.** Metrics are written to a JSON file, not logged to a tracker.

## Layout

```
src/telco/data.py          IBM churn loader; synthetic fraud generator
src/telco/models.py        churn and fraud pipelines, metrics, threshold selection
src/telco/rag.py           chunking, TF-IDF and LSA retrieval, retrieval evaluation
src/telco/services.py      builds trained artifacts once; tool implementations
src/telco/agent.py         tool-using agent (LLM mode and offline mode)
src/telco/eval_agent.py    agent evaluation harness
src/telco/api.py           FastAPI app
src/telco/cli.py           train and report
data/raw/                  IBM churn CSV
docs/policies/             illustrative policy documents for RAG
docs/eval_questions.json   labelled retrieval questions
docs/agent_eval.json       labelled agent cases
tests/                     pytest suite
.github/workflows/ci.yml   tests and pipeline run on every push
Dockerfile                 container for the API
```
