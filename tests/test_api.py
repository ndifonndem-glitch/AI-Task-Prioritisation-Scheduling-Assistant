import pytest
from fastapi.testclient import TestClient
from ml_service.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "DecisionTreeClassifier" in data["model"]
    assert "llm_backends" in data
    assert data["llm_backends"]["base_router_port_8080"] == "UP"
    assert data["llm_backends"]["fine_tuned_router_port_8081"] == "UP"

def test_predict_valid_high_priority():
    payload = {
        "deadline_hours": 8,
        "duration_hours": 3,
        "importance": 5,
        "dependency": 1,
        "available_hours": 6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["priority"] in ["Do now", "Schedule this week", "Defer"]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["calendar_slot_feasible"] is True
    assert "feature_contributions" in data
    assert "guardrails_applied" in data
    assert isinstance(data["guardrails_applied"], list)

def test_predict_guardrails_applied_blocked_dependency():
    payload = {
        "deadline_hours": 8,
        "duration_hours": 3,
        "importance": 5,
        "dependency": 1,
        "available_hours": 6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["guardrails_applied"]) > 0
    assert any("Blocked Dependency" in g for g in data["guardrails_applied"])

def test_predict_valid_low_priority():
    payload = {
        "deadline_hours": 168,
        "duration_hours": 1,
        "importance": 1,
        "dependency": 0,
        "available_hours": 8
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["priority"] in ["Defer", "Schedule this week"]
    assert data["calendar_slot_feasible"] is True
    assert isinstance(data["guardrails_applied"], list)

def test_predict_infeasible_slot():
    payload = {
        "deadline_hours": 12,
        "duration_hours": 6,
        "importance": 4,
        "dependency": 1,
        "available_hours": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["calendar_slot_feasible"] is False
    assert any("Infeasible Calendar Slot" in g for g in data["guardrails_applied"])

def test_predict_malformed_negative_duration():
    payload = {
        "deadline_hours": 12,
        "duration_hours": -4,
        "importance": 5,
        "dependency": 1,
        "available_hours": 4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Invalid input"
    assert "duration_hours" in data["details"]

def test_predict_malformed_importance_out_of_range():
    payload = {
        "deadline_hours": 12,
        "duration_hours": 3,
        "importance": 10,
        "dependency": 1,
        "available_hours": 4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Invalid input"
    assert "importance" in data["details"]

def test_predict_malformed_invalid_data_type():
    payload = {
        "deadline_hours": "tomorrow",
        "duration_hours": 3,
        "importance": 5,
        "dependency": 1,
        "available_hours": 4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Invalid input"

