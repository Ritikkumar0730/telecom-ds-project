"""Evaluate the offline agent on labelled cases: did it pick the right tool and cite the right text?

Scores tool-selection accuracy and an answer-content check. Run with the LLM client too by
passing a client to Agent; the offline router is the deterministic baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

from telco.agent import Agent
from telco.services import Services

CASES = Path(__file__).resolve().parents[2] / "docs" / "agent_eval.json"


def evaluate_agent(agent: Agent, cases: list[dict]) -> dict:
    tool_ok = content_ok = 0
    failures = []
    for c in cases:
        out = agent.answer(c["question"])
        tools = [t["tool"] for t in out["tool_calls"]]
        first = tools[0] if tools else None
        t_ok = first == c["expect_tool"]
        a_ok = c["answer_contains"].lower() in out["answer"].lower()
        tool_ok += t_ok
        content_ok += a_ok
        if not (t_ok and a_ok):
            failures.append({"question": c["question"], "tools": tools, "answer": out["answer"][:200]})
    n = len(cases)
    return {
        "n_cases": n,
        "tool_selection_accuracy": round(tool_ok / n, 4),
        "answer_content_accuracy": round(content_ok / n, 4),
        "failures": failures,
    }


if __name__ == "__main__":
    services = Services.build()
    result = evaluate_agent(Agent(services, client=None), json.loads(CASES.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2))
