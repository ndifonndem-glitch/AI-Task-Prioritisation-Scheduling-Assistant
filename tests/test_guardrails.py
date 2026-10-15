import pytest
from agent.guardrails import GuardrailEngine

@pytest.fixture
def engine():
    return GuardrailEngine()

def test_guardrail_invalid_negative_duration(engine):
    data = {"deadline_hours": 12, "duration_hours": -2, "importance": 4, "available_hours": 4}
    valid, msg = engine.validate_input(data)
    assert valid is False
    assert "duration_hours" in msg

def test_guardrail_invalid_importance(engine):
    data = {"deadline_hours": 12, "duration_hours": 3, "importance": 10, "available_hours": 4}
    valid, msg = engine.validate_input(data)
    assert valid is False
    assert "importance" in msg

def test_guardrail_api_failure_handling(engine):
    empty_res = {}
    valid, msg = engine.check_api_reliability(empty_res)
    assert valid is False
    assert "unavailable" in msg

def test_guardrail_human_approval_protected_branch(engine):
    requires_approval, msg = engine.evaluate_human_approval_boundary(
        action="delete", priority="MEDIUM", branch="main"
    )
    assert requires_approval is True
    assert "HUMAN APPROVAL REQUIRED" in msg

def test_guardrail_human_approval_displace_high_priority(engine):
    requires_approval, msg = engine.evaluate_human_approval_boundary(
        action="displace_task", priority="HIGH", branch="feature"
    )
    assert requires_approval is True
    assert "HUMAN APPROVAL REQUIRED" in msg

def test_guardrail_preapproved_low_risk_action(engine):
    requires_approval, msg = engine.evaluate_human_approval_boundary(
        action="ADD_SCHEDULED_TASK", priority="LOW", branch="feature"
    )
    assert requires_approval is False
    assert "pre-approved" in msg
