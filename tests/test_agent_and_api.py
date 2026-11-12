from fastapi.testclient import TestClient

from telco.agent import Agent


def test_offline_agent_routes_customer_question(services):
    agent = Agent(services, client=None)
    out = agent.answer("What is the churn risk for customer 7590-VHVEG?")
    assert out["mode"] == "offline"
    tools = [t["tool"] for t in out["tool_calls"]]
    assert tools[0] == "get_churn_risk"
    assert "churn probability" in out["answer"]


def test_offline_agent_asks_for_missing_fraud_fields(services):
    agent = Agent(services, client=None)
    out = agent.answer("Is this transaction fraud?")
    assert "I need" in out["answer"]
    assert not any(t["tool"] == "check_transaction_fraud" for t in out["tool_calls"])


def test_offline_agent_scores_fraud_when_fields_given(services):
    agent = Agent(services, client=None)
    q = ("Check fraud for this app transaction amount=18000 hour=2 velocity_1h=5 "
         "new_device=1 distance_km=250 account_age_days=20 international=1")
    out = agent.answer(q)
    assert any(t["tool"] == "check_transaction_fraud" for t in out["tool_calls"])
    assert "Fraud probability" in out["answer"]


def test_offline_agent_policy_answer_cites_doc(services):
    agent = Agent(services, client=None)
    out = agent.answer("How long is a prepaid pack valid after activation?")
    assert "prepaid_plans" in out["answer"]


class _FakeBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeResp:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _FakeResp(
                "tool_use",
                [_FakeBlock(type="tool_use", id="t1", name="get_churn_risk", input={"customer_id": "7590-VHVEG"})],
            )
        return _FakeResp("end_turn", [_FakeBlock(type="text", text="Customer 7590-VHVEG is at low risk.")])


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_llm_mode_runs_tool_loop_with_client(services):
    agent = Agent(services, client=_FakeClient())
    out = agent.answer("Churn risk for 7590-VHVEG?")
    assert out["mode"] == "llm"
    assert out["tool_calls"][0]["tool"] == "get_churn_risk"
    assert out["answer"] == "Customer 7590-VHVEG is at low risk."


def test_api_endpoints(services, monkeypatch):
    from telco import api

    monkeypatch.setitem(api.state, "services", services)
    monkeypatch.setitem(api.state, "agent", Agent(services, client=None))
    # Bypass lifespan (which rebuilds models) by using the client without context manager.
    client = TestClient(api.app)

    assert client.get("/customers/C999999/churn-risk").status_code == 404

    ok = client.post(
        "/transactions/fraud-score",
        json={"amount": 500, "hour": 14, "velocity_1h": 1, "new_device": 0,
              "distance_km": 3, "account_age_days": 900, "international": 0, "channel": "app"},
    )
    assert ok.status_code == 200 and "fraud_probability" in ok.json()

    bad = client.post(
        "/transactions/fraud-score",
        json={"amount": -5, "hour": 30, "velocity_1h": 1, "new_device": 0,
              "distance_km": 3, "account_age_days": 900, "international": 0, "channel": "app"},
    )
    assert bad.status_code == 422

    res = client.post("/policy/search", json={"query": "refund for duplicate charge", "k": 2})
    assert res.status_code == 200 and res.json()["results"][0]["doc_id"] == "billing_and_refunds"


def test_agent_eval_harness_passes_offline_baseline(services):
    import json
    from pathlib import Path

    from telco.eval_agent import evaluate_agent

    cases = json.loads((Path(__file__).resolve().parents[1] / "docs" / "agent_eval.json").read_text())
    result = evaluate_agent(Agent(services, client=None), cases)
    assert result["tool_selection_accuracy"] == 1.0, result["failures"]
