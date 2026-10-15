import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent.tools.gitea_tool import GiteaRepositoryTool

def test_gitea_action_and_independent_verification():
    print("=" * 70)
    print("OPENCLAW -> GITEA ACTION & INDEPENDENT VERIFICATION TEST")
    print("=" * 70)

    gitea_tool = GiteaRepositoryTool()

    # 1. Execute controlled Gitea Action
    action_res = gitea_tool.execute_gitea_action(
        task_name="Deploy critical security patch",
        priority="Do now",
        probability=0.9850,
        explanation="Task deadline is within 2 hours with high stated importance."
    )

    commit_id = action_res["commit_id"]
    print(f"Executed Gitea Action. Generated Commit ID: '{commit_id}'")

    # 2. Independent Verification (Queries file system directly without agent self-report)
    is_verified, message = gitea_tool.verify_gitea_action(commit_id)

    print(f"Independent Verification Result: {is_verified}")
    print(f"Verification Detail Message: '{message}'")

    assert is_verified is True, f"Independent verification failed: {message}"
    print("-" * 70)
    print("VERIFICATION PASSED: Repository changes confirmed present independently.")
    print("=" * 70)
    return True

if __name__ == "__main__":
    test_gitea_action_and_independent_verification()
