from typing import Dict, Any, Tuple

class GuardrailEngine:
    def __init__(self, protected_branches: list = None):
        self.protected_branches = protected_branches or ["main", "master", "production"]

    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Guardrail Failure Mode 1: Validation of prediction input parameters.
        Prevents invalid/corrupt data from reaching the prediction API.
        """
        deadline = data.get("deadline_hours")
        duration = data.get("duration_hours")
        importance = data.get("importance")
        available = data.get("available_hours")

        if deadline is not None and deadline < 0:
            return False, "Guardrail Triggered: deadline_hours cannot be negative."
        if duration is not None and duration <= 0:
            return False, "Guardrail Triggered: duration_hours must be greater than 0."
        if importance is not None and (importance < 1 or importance > 5):
            return False, "Guardrail Triggered: importance must be between 1 and 5."
        if available is not None and available < 0:
            return False, "Guardrail Triggered: available_hours cannot be negative."

        return True, "Input validation passed."

    def check_api_reliability(self, api_response: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Guardrail Failure Mode 2: Prediction API failure handling.
        Prevents the agent from fabricating predictions if the ML service fails.
        """
        if not api_response or "error" in api_response:
            err_msg = api_response.get("error", "ML Prediction service returned empty response.")
            return False, f"Guardrail Triggered: Prediction service unavailable ({err_msg}). Agent will not fabricate predictions."
        return True, "API response reliable."

    def evaluate_human_approval_boundary(self, action: str, priority: str, branch: str = "main") -> Tuple[bool, str]:
        """
        Guardrail Failure Mode 3: Human Approval Boundary.
        Actions modifying protected content or displacing HIGH priority tasks require explicit human approval.
        """
        if branch in self.protected_branches and action in ["delete", "overwrite", "force_push"]:
            return True, f"HUMAN APPROVAL REQUIRED: Action '{action}' on protected branch '{branch}' requires human approval."
        
        if action in ["displace_task", "displace_existing_urgent_task", "cancel_task"] or (priority in ["HIGH", "Do now"] and action in ["displace_task", "cancel_task"]):
            return True, f"HUMAN APPROVAL REQUIRED: Displacing or cancelling an existing task requires explicit human approval."

        return False, "Action pre-approved for execution."
