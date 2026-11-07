import os
import requests
from typing import Dict, Any

# The agent must call the prediction model as a TOOL, over HTTP. An in-process
# fallback that imports the predictor directly defeats that: it makes the agent and
# the model one program again, and it silently suppresses the API-offline guardrail,
# because a request that cannot reach the service still returns a prediction.
#
# The fallback is therefore disabled by default and available only to unit tests that
# deliberately ask for it. When it is used, the response is labelled so no caller can
# mistake an in-process result for a service response.
ALLOW_IN_PROCESS = os.environ.get("PREDICT_ALLOW_IN_PROCESS", "").lower() in ("1", "true", "yes")


class PredictionServiceTool:
    def __init__(self, endpoint_url: str = None):
        self.endpoint_url = endpoint_url or os.environ.get(
            "PREDICT_URL", "http://127.0.0.1:5000/predict")

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
            response = requests.post(self.endpoint_url, json=payload, timeout=3.0)
        except Exception as ex:
            # Terminal. The service is unreachable, so no prediction exists. Returning an
            # "error" key is what causes guardrails.check_api_reliability to trigger
            # FAILED_API_GUARDRAIL; returning a locally computed answer here would hide it.
            if ALLOW_IN_PROCESS:
                return self._in_process(payload, reason=f"{type(ex).__name__}: {ex}")
            return {
                "error": "Prediction service unreachable",
                "details": f"{type(ex).__name__}: {ex}",
                "endpoint": self.endpoint_url,
                "guidance": ("No prediction was made. Do not treat this as a low-priority "
                             "result and do not substitute a locally computed value."),
            }

        if response.status_code == 200:
            return response.json()

        # A 4xx or 5xx is also an absence of a prediction, not a quiet one.
        return {
            "error": f"HTTP {response.status_code}",
            "details": response.text[:400],
            "endpoint": self.endpoint_url,
            "guidance": "No prediction was made. The request did not succeed.",
        }

    # ------------------------------------------------------------------ test-only path
    def _in_process(self, payload: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Offline unit-test path. Never reached unless PREDICT_ALLOW_IN_PROCESS is set."""
        from ml_service.predictor import predictor_instance
        try:
            p, prob, probs, feats, fit, g_applied = predictor_instance.predict(
                deadline_hours=payload["deadline_hours"],
                duration_hours=payload["duration_hours"],
                importance=payload["importance"],
                dependency=payload["dependency"],
                available_hours=payload["available_hours"],
            )
            return {
                "priority": p,
                "probability": prob,
                "probabilities": probs,
                "feature_contributions": feats,
                "calendar_slot_feasible": fit,
                "guardrails_applied": list(g_applied) + ["in_process_fallback"],
                "source": "IN-PROCESS FALLBACK",
                "warning": ("Computed in-process because the HTTP service was unreachable "
                            f"({reason}). This did not exercise the prediction service and "
                            "must not be presented as evidence of a working tool call."),
            }
        except Exception as ex:
            return {"error": "Prediction exception", "details": str(ex)}
