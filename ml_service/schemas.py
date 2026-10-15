from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class PredictionRequest(BaseModel):
    deadline_hours: float = Field(..., description="Hours remaining before task deadline (must be >= 0)", ge=0.0)
    duration_hours: float = Field(..., description="Estimated completion time in hours (must be > 0)", gt=0.0)
    importance: int = Field(..., description="Task importance score between 1 and 5", ge=1, le=5)
    dependency: int = Field(..., description="1 if task depends on another task, else 0", ge=0, le=1)
    available_hours: float = Field(..., description="User's currently available working hours (must be >= 0)", ge=0.0)
    n_dependents: Optional[int] = Field(default=0, description="Number of downstream tasks dependent on this task", ge=0)
    times_deferred: Optional[int] = Field(default=0, description="Number of times task was previously deferred", ge=0)
    age_days: Optional[int] = Field(default=0, description="Task age in days", ge=0)

class PredictionResponse(BaseModel):
    priority: str
    probability: float
    probabilities: Dict[str, float]
    feature_contributions: Dict[str, float]
    calendar_slot_feasible: bool
    guardrails_applied: List[str]
