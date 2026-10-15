import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "decision_tree_priority.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

class TaskPriorityPredictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
            
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        self.feature_names = self.metadata["feature_names"]
        self.classes = list(self.model.classes_)

    def predict(
        self,
        deadline_hours: float,
        duration_hours: float,
        importance: int,
        dependency: int,
        available_hours: float,
        n_dependents: int = 0,
        times_deferred: int = 0,
        age_days: int = 0
    ) -> Tuple[str, float, Dict[str, float], Dict[str, float], bool, List[str]]:

        # Map API input parameters to model feature schema
        hours_to_deadline = float(deadline_hours)
        est_duration_h = float(duration_hours)
        stated_importance = int(importance)
        dep_blocked = int(dependency)
        slack_ratio = hours_to_deadline / (est_duration_h + 0.001)
        fits_largest_block = 1 if est_duration_h <= available_hours else 0

        urgency_density = 1.0 / (hours_to_deadline + 0.1)
        deferral_starvation_index = float(times_deferred * (age_days + 1.0))
        work_pressure_ratio = est_duration_h / (hours_to_deadline + 0.1)

        input_dict = {
            "hours_to_deadline": hours_to_deadline,
            "est_duration_h": est_duration_h,
            "stated_importance": stated_importance,
            "slack_ratio": slack_ratio,
            "fits_largest_block": fits_largest_block,
            "dep_blocked": dep_blocked,
            "n_dependents": n_dependents,
            "times_deferred": times_deferred,
            "age_days": age_days,
            "urgency_density": urgency_density,
            "deferral_starvation_index": deferral_starvation_index,
            "work_pressure_ratio": work_pressure_ratio
        }

        # Format input for scikit-learn
        X_input = pd.DataFrame([input_dict])[self.feature_names]

        # Prediction
        prediction_class = str(self.model.predict(X_input)[0])
        probabilities_array = self.model.predict_proba(X_input)[0]

        prob_dict = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.classes, probabilities_array)
        }

        predicted_prob = prob_dict.get(prediction_class, round(float(np.max(probabilities_array)), 4))

        # Feasibility check
        calendar_slot_feasible = duration_hours <= available_hours

        # Explicit Guardrail Traceability Assessment 2 Recording
        guardrails_applied = []
        if dep_blocked == 1:
            if prediction_class in ["HIGH", "Do now"]:
                prediction_class = "Schedule this week"
            guardrails_applied.append("blocked_task_never_do_now")
            guardrails_applied.append("Guardrail Applied: Blocked Dependency (dep_blocked=1) - Priority capped to Defer/Schedule")

        if not calendar_slot_feasible:
            guardrails_applied.append("infeasible_calendar_slot")
            guardrails_applied.append("Guardrail Applied: Infeasible Calendar Slot (duration > available_hours)")


        # Feature contributions
        feature_importances = self.metadata.get("feature_importances", {})

        return prediction_class, predicted_prob, prob_dict, feature_importances, calendar_slot_feasible, guardrails_applied

# Singleton instance
predictor_instance = TaskPriorityPredictor()
