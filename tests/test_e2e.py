import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from agent.openclaw_agent import OpenClawTaskAssistantAgent

BASE_DIR = Path(__file__).parent.parent
E2E_RESULTS_FILE = BASE_DIR / "docs" / "e2e_evaluation_results.json"
E2E_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

# 10 Test Scenarios for End-to-End Repeated Evaluation
TEST_SCENARIOS = [
    {
        "id": 1,
        "description": "Complete client financial report",
        "info": {"deadline_hours": 12, "duration_hours": 3, "importance": 5, "dependency": 1, "available_hours": 6},
        "expected_status": "SUCCESS"
    },
    {
        "id": 2,
        "description": "Routine documentation update",
        "info": {"deadline_hours": 168, "duration_hours": 1, "importance": 1, "dependency": 0, "available_hours": 8},
        "expected_status": "SUCCESS"
    },
    {
        "id": 3,
        "description": "Database migration prep",
        "info": {"deadline_hours": 24, "duration_hours": 4, "importance": 4, "dependency": 1, "available_hours": 5},
        "expected_status": "SUCCESS"
    },
    {
        "id": 4,
        "description": "Malformed negative duration task",
        "info": {"deadline_hours": 24, "duration_hours": -2, "importance": 4, "dependency": 0, "available_hours": 5},
        "expected_status": "REJECTED_GUARDRAIL"
    },
    {
        "id": 5,
        "description": "Displace existing urgent task",
        "info": {"deadline_hours": 4, "duration_hours": 2, "importance": 5, "dependency": 1, "available_hours": 3},
        "action": "displace_task",
        "expected_status": "PAUSED_REQUIRES_HUMAN_APPROVAL"
    },
    {
        "id": 6,
        "description": "Marketing presentation review",
        "info": {"deadline_hours": 48, "duration_hours": 2, "importance": 3, "dependency": 0, "available_hours": 4},
        "expected_status": "SUCCESS"
    },
    {
        "id": 7,
        "description": "Security audit log review",
        "info": {"deadline_hours": 8, "duration_hours": 3, "importance": 5, "dependency": 0, "available_hours": 4},
        "expected_status": "SUCCESS"
    },
    {
        "id": 8,
        "description": "Invalid importance rating task",
        "info": {"deadline_hours": 24, "duration_hours": 2, "importance": 99, "dependency": 0, "available_hours": 4},
        "expected_status": "REJECTED_GUARDRAIL"
    },
    {
        "id": 9,
        "description": "Website bug fix deploy",
        "info": {"deadline_hours": 16, "duration_hours": 2, "importance": 4, "dependency": 1, "available_hours": 4},
        "expected_status": "SUCCESS"
    },
    {
        "id": 10,
        "description": "Delete core task on main branch",
        "info": {"deadline_hours": 72, "duration_hours": 1, "importance": 2, "dependency": 0, "available_hours": 4},
        "action": "delete",
        "branch": "main",
        "expected_status": "PAUSED_REQUIRES_HUMAN_APPROVAL"
    }
]

def run_e2e_evaluation():
    print("=" * 70)
    print("RUNNING END-TO-END SYSTEM REPEATED EVALUATION (10 RUNS)")
    print("=" * 70)

    agent = OpenClawTaskAssistantAgent()
    run_records = []
    pass_count = 0

    for sc in TEST_SCENARIOS:
        action = sc.get("action", "ADD_SCHEDULED_TASK")
        branch = sc.get("branch", "feature")

        result = agent.process_task_request(
            task_description=sc["description"],
            extracted_info=sc["info"],
            requested_action=action,
            target_branch=branch
        )

        status_received = result["status"]
        passed = (status_received == sc["expected_status"])
        if passed: pass_count += 1

        record = {
            "run_id": sc["id"],
            "description": sc["description"],
            "expected_status": sc["expected_status"],
            "actual_status": status_received,
            "passed": passed,
            "prediction": result.get("prediction", {}).get("priority", "N/A"),
            "independent_verification": result.get("independent_verification", {}).get("verified", False)
        }
        run_records.append(record)

        print(f"Run {sc['id']:02d}: {sc['description']:<35} | Expected: {sc['expected_status']:<30} | Result: {'PASS' if passed else 'FAIL'}")

    summary = {
        "total_runs": len(TEST_SCENARIOS),
        "passed_runs": pass_count,
        "pass_rate_pct": round((pass_count / len(TEST_SCENARIOS)) * 100, 2),
        "run_details": run_records
    }

    with open(E2E_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"E2E EVALUATION SUMMARY: {pass_count}/{len(TEST_SCENARIOS)} PASSED ({summary['pass_rate_pct']}%)")
    print(f"Saved evaluation artifact to {E2E_RESULTS_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    run_e2e_evaluation()
