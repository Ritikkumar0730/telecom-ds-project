"""Tool-using agent over the churn model, fraud model and policy RAG.

Two modes:
- LLM mode (ANTHROPIC_API_KEY set): Claude chooses tools via the Messages tool-use API and
  writes the final answer grounded in tool results. Loop is capped at MAX_STEPS.
- Offline mode (no key): a deterministic router picks the same tools from the question text,
  so the pipeline runs and is testable without network access or cost.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from telco.services import Services

MAX_STEPS = 5
DEFAULT_MODEL = os.environ.get("TELCO_LLM_MODEL", "claude-sonnet-5-5")

TOOLS = [
    {
        "name": "get_churn_risk",
        "description": "Predicted churn probability and key attributes for one customer id, e.g. 7590-VHVEG.",
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "check_transaction_fraud",
        "description": "Fraud probability for a single payment transaction. Amount in INR, hour 0-23.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "hour": {"type": "integer"},
                "velocity_1h": {"type": "integer"},
                "new_device": {"type": "integer", "enum": [0, 1]},
                "distance_km": {"type": "number"},
                "account_age_days": {"type": "integer"},
                "international": {"type": "integer", "enum": [0, 1]},
                "channel": {"type": "string", "enum": ["app", "web", "pos"]},
            },
            "required": ["amount", "hour", "velocity_1h", "new_device", "distance_km", "account_age_days", "international", "channel"],
        },
    },
    {
        "name": "search_policy",
        "description": "Search internal plan, billing, retention and fraud policy documents. Returns cited passages.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

SYSTEM_PROMPT = (
    "You are an assistant for a telecom operations team. Use the tools for any customer-specific "
    "or policy question. Cite the policy doc_id and section you used. Never invent customer data. "
    "If a fraud check lacks required transaction fields, ask for them instead of guessing."
)


class Agent:
    def __init__(self, services: Services, client: Any | None = None, model: str = DEFAULT_MODEL):
        self.services = services
        self.client = client
        self.model = model

    @property
    def mode(self) -> str:
        return "llm" if self.client is not None else "offline"

    def run_tool(self, name: str, args: dict) -> dict:
        if name == "get_churn_risk":
            return self.services.churn_risk(args["customer_id"])
        if name == "check_transaction_fraud":
            return self.services.fraud_score(args)
        if name == "search_policy":
            return self.services.search_policy(args["query"])
        return {"error": f"unknown tool {name}"}

    def answer(self, question: str) -> dict:
        if self.client is not None:
            return self._answer_llm(question)
        return self._answer_offline(question)

    # ----- LLM mode -----
    def _answer_llm(self, question: str) -> dict:
        messages: list[dict] = [{"role": "user", "content": question}]
        trace: list[dict] = []
        for _ in range(MAX_STEPS):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                return {"answer": text, "tool_calls": trace, "mode": "llm"}
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if getattr(block, "type", "") == "tool_use":
                    out = self.run_tool(block.name, block.input)
                    trace.append({"tool": block.name, "input": block.input, "output": out})
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out)}
                    )
            messages.append({"role": "user", "content": results})
        return {"answer": "Stopped: tool-call limit reached.", "tool_calls": trace, "mode": "llm"}

    # ----- Offline mode -----
    def _answer_offline(self, question: str) -> dict:
        q = question.lower()
        trace: list[dict] = []
        parts: list[str] = []

        cust = re.search(r"\b\d{4}-[A-Z]{5}\b", question)
        if cust:
            out = self.run_tool("get_churn_risk", {"customer_id": cust.group(0)})
            trace.append({"tool": "get_churn_risk", "input": {"customer_id": cust.group(0)}, "output": out})
            if "error" in out:
                parts.append(out["error"])
            else:
                level = "high" if out["high_risk"] else "low"
                parts.append(
                    f"{out['customer_id']} has a {out['churn_probability']:.1%} predicted churn "
                    f"probability ({level} risk at threshold {self.services.churn.threshold})."
                )

        if "fraud" in q or "transaction" in q:
            nums = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", question)}
            needed = {"amount", "hour", "velocity_1h", "new_device", "distance_km", "account_age_days", "international"}
            if needed <= nums.keys():
                args = {
                    "amount": nums["amount"],
                    "hour": int(nums["hour"]),
                    "velocity_1h": int(nums["velocity_1h"]),
                    "new_device": int(nums["new_device"]),
                    "distance_km": nums["distance_km"],
                    "account_age_days": int(nums["account_age_days"]),
                    "international": int(nums["international"]),
                    "channel": "app",
                }
                for ch in ("app", "web", "pos"):
                    if re.search(rf"\b{ch}\b", q):
                        args["channel"] = ch
                out = self.run_tool("check_transaction_fraud", args)
                trace.append({"tool": "check_transaction_fraud", "input": args, "output": out})
                verdict = "flagged for review" if out["flagged"] else "not flagged"
                parts.append(f"Fraud probability {out['fraud_probability']:.2%}; transaction is {verdict}.")
            else:
                parts.append(
                    "To check a transaction I need: amount, hour, velocity_1h, new_device, distance_km, "
                    "account_age_days, international (and optionally channel)."
                )

        if not parts or "policy" in q or "how" in q or "what" in q or "can " in q:
            out = self.run_tool("search_policy", {"query": question})
            trace.append({"tool": "search_policy", "input": {"query": question}, "output": out})
            top = out["results"][:1]
            if top:
                c = top[0]
                parts.append(f"From {c['doc_id']} / {c['section']}: {c['text']}")
            else:
                parts.append("No matching policy passage found.")

        return {"answer": " ".join(parts), "tool_calls": trace, "mode": "offline"}


def make_agent(services: Services) -> Agent:
    """Use Claude when the key and SDK are available; otherwise fall back to offline routing."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            return Agent(services, client=anthropic.Anthropic())
        except Exception:
            pass
    return Agent(services, client=None)
