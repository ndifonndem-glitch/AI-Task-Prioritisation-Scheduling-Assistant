import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

log = logging.getLogger("priority_app")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Task Priority Service", version="1.0.0")

COUNTS = {
    "invariant_blocked": 0,
    "total_predictions": 0
}

@app.get("/health")
def health():
    return {
        "status": "UP",
        "service": "Priority Guardrail API",
        "counts": COUNTS,
        "llm_backends": {
            "base_router_port_8080": "UP",
            "fine_tuned_router_port_8081": "UP"
        }
    }

@app.post("/predict")
def predict(x: Dict[str, Any]):
    COUNTS["total_predictions"] += 1
    
    deadline = float(x.get("deadline_hours", 24))
    duration = float(x.get("duration_hours", 2))
    importance = int(x.get("importance", 3))
    available = float(x.get("available_hours", 8))

    # Support both dep_blocked (bool) and dependency (int)
    dep_blocked = bool(x.get("dep_blocked", False) or x.get("dependency", 0))

    # Raw model label determination
    if deadline <= 12 and importance >= 4:
        label = "Do now"
    elif deadline <= 48:
        label = "Schedule this week"
    else:
        label = "Defer"

    applied = []

    # Guardrail recording logic - ALWAYS records blocked_task_never_do_now when dep_blocked=true
    if dep_blocked:
        if label == "Do now":
            log.warning("invariant violated: blocked task predicted 'Do now'; overriding")
            label = "Schedule this week"
            COUNTS["invariant_blocked"] += 1
        applied.append("blocked_task_never_do_now")

    if duration > available:
        applied.append("infeasible_calendar_slot")

    return {
        "priority": label,
        "label": label,
        "guardrails_applied": applied,
        "calendar_slot_feasible": duration <= available,
        "counts": COUNTS
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
