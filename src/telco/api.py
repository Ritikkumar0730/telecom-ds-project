"""FastAPI service exposing churn scoring, fraud scoring, policy search and the agent."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from telco.agent import make_agent
from telco.services import Services

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = Services.build()
    state["services"] = services
    state["agent"] = make_agent(services)
    yield
    state.clear()


app = FastAPI(title="Telecom DS: churn, fraud, RAG and agent", lifespan=lifespan)


class FraudRequest(BaseModel):
    amount: float = Field(gt=0)
    hour: int = Field(ge=0, le=23)
    velocity_1h: int = Field(ge=0)
    new_device: int = Field(ge=0, le=1)
    distance_km: float = Field(ge=0)
    account_age_days: int = Field(ge=0)
    international: int = Field(ge=0, le=1)
    channel: str = Field(pattern="^(app|web|pos)$")


class SearchRequest(BaseModel):
    query: str = Field(min_length=3)
    k: int = Field(default=3, ge=1, le=10)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@app.get("/health")
def health():
    s: Services = state["services"]
    return {
        "status": "ok",
        "agent_mode": state["agent"].mode,
        "churn_metrics": s.churn.metrics,
        "fraud_metrics": s.fraud.metrics,
        "rag_metrics": {k: v for k, v in s.rag_metrics.items() if k != "misses"},
    }


@app.get("/customers/{customer_id}/churn-risk")
def churn_risk(customer_id: str):
    out = state["services"].churn_risk(customer_id)
    if "error" in out:
        raise HTTPException(status_code=404, detail=out["error"])
    return out


@app.post("/transactions/fraud-score")
def fraud_score(req: FraudRequest):
    return state["services"].fraud_score(req.model_dump())


@app.post("/policy/search")
def policy_search(req: SearchRequest):
    return state["services"].search_policy(req.query, k=req.k)


@app.post("/ask")
def ask(req: AskRequest):
    return state["agent"].answer(req.question)
