import requests
from typing import Dict, Any

class PredictionServiceTool:
    def __init__(self, endpoint_url: str = "http://127.0.0.1:5000/predict"):
        self.endpoint_url = endpoint_url

    def call_predict(
        self,
        deadline_hours: float,
        duration_hours: float,
        importance: int,
        dependency: int,
        available_hours: float
    ) -> Dict[str, Any]:
        
        payload = {
            "deadline_hours": deadline_hours,
            "duration_hours": duration_hours,
            "importance": importance,
            "dependency": dependency,
            "available_hours": available_hours
        }

        try:
            # First try external server endpoint
            response = requests.post(self.endpoint_url, json=payload, timeout=3.0)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}", "details": response.text}
        except Exception:
            # Fallback to direct predictor instance in same process if web server is not running locally
            from ml_service.predictor import predictor_instance
            try:
                p, prob, probs, feats, fit, g_applied = predictor_instance.predict(
                    deadline_hours=deadline_hours,
                    duration_hours=duration_hours,
                    importance=importance,
                    dependency=dependency,
                    available_hours=available_hours
                )
                return {
                    "priority": p,
                    "probability": prob,
                    "probabilities": probs,
                    "feature_contributions": feats,
                    "calendar_slot_feasible": fit,
                    "guardrails_applied": g_applied
                }
            except Exception as ex:
                return {"error": "Prediction exception", "details": str(ex)}

