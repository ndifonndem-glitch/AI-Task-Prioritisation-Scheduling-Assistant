import json
from typing import Dict, Any
from agent.guardrails import GuardrailEngine
from agent.tools.prediction_tool import PredictionServiceTool
from agent.tools.gitea_tool import GiteaRepositoryTool

class OpenClawTaskAssistantAgent:
    def __init__(self):
        self.guardrails = GuardrailEngine()
        self.predictor_tool = PredictionServiceTool()
        self.gitea_tool = GiteaRepositoryTool()

    def process_task_request(
        self,
        task_description: str,
        extracted_info: Dict[str, Any],
        requested_action: str = "ADD_SCHEDULED_TASK",
        target_branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Main OpenClaw Agent Orchestration Loop:
        1. Receive user task request.
        2. Apply Guardrail 1 (Input Validation).
        3. Call Prediction Tool (Component A).
        4. Apply Guardrail 2 (API Reliability Check).
        5. Apply Guardrail 3 (Human Approval Boundary Check).
        6. Execute Gitea action (Component C).
        7. Execute Independent Verification.
        8. STOP and return result.
        """
        agent_log = []
        agent_log.append(f"Agent Goal: Process task '{task_description}' and record approved schedule in Gitea.")

        # Step 1: Input Validation Guardrail
        valid_input, msg_input = self.guardrails.validate_input(extracted_info)
        agent_log.append(f"Step 1 Guardrail Check: {msg_input}")
        if not valid_input:
            return {
                "status": "REJECTED_GUARDRAIL",
                "reason": msg_input,
                "agent_log": agent_log
            }

        # Step 2: Call Prediction API Tool
        pred_result = self.predictor_tool.call_predict(
            deadline_hours=extracted_info["deadline_hours"],
            duration_hours=extracted_info["duration_hours"],
            importance=extracted_info["importance"],
            dependency=extracted_info["dependency"],
            available_hours=extracted_info["available_hours"]
        )

        # Step 3: API Reliability Guardrail
        valid_api, msg_api = self.guardrails.check_api_reliability(pred_result)
        agent_log.append(f"Step 2 API Check: {msg_api}")
        if not valid_api:
            return {
                "status": "FAILED_API_GUARDRAIL",
                "reason": msg_api,
                "agent_log": agent_log
            }

        priority = pred_result["priority"]
        probability = pred_result["probability"]
        feasible = pred_result["calendar_slot_feasible"]

        # Step 4: Human Approval Boundary Check
        requires_approval, approval_msg = self.guardrails.evaluate_human_approval_boundary(
            action=requested_action,
            priority=priority,
            branch=target_branch
        )
        agent_log.append(f"Step 3 Approval Check: {approval_msg}")
        if requires_approval:
            return {
                "status": "PAUSED_REQUIRES_HUMAN_APPROVAL",
                "reason": approval_msg,
                "prediction": pred_result,
                "agent_log": agent_log
            }

        # Step 5: Format Human-Readable Explanation
        explanation = (
            f"The task '{task_description}' was predicted as {priority} priority (confidence: {probability * 100:.1f}%). "
            f"Calendar slot feasibility: {'FEASIBLE' if feasible else 'INFEASIBLE (exceeds available hours)'}."
        )

        # Step 6: Perform Gitea Action
        gitea_res = self.gitea_tool.execute_gitea_action(
            task_name=task_description,
            priority=priority,
            probability=probability,
            explanation=explanation
        )
        agent_log.append(f"Step 4 Gitea Commit Executed: Commit ID '{gitea_res['commit_id']}'")

        # Step 7: Independent Verification (Not trusting agent self-report)
        verified, ver_msg = self.gitea_tool.verify_gitea_action(gitea_res["commit_id"])
        agent_log.append(f"Step 5 Independent Verification: {ver_msg}")

        # Step 8: Stopping Condition Reached
        agent_log.append("Agent Stopping Condition Reached: Action verified and complete.")

        return {
            "status": "SUCCESS" if verified else "FAILED_VERIFICATION",
            "task_description": task_description,
            "prediction": pred_result,
            "explanation": explanation,
            "gitea_result": gitea_res,
            "independent_verification": {"verified": verified, "message": ver_msg},
            "agent_log": agent_log
        }
