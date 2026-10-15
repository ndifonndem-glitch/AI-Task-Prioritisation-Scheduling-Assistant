import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

class GiteaRepositoryTool:
    def __init__(self, repo_dir: Path = None):
        if repo_dir is None:
            repo_dir = Path(__file__).parent.parent.parent / "data" / "gitea_mock_repo"
        self.repo_dir = repo_dir
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_file = self.repo_dir / "TASKS.md"
        self.log_file = self.repo_dir / "commit_log.json"

    def execute_gitea_action(self, task_name: str, priority: str, probability: float, explanation: str) -> Dict[str, Any]:
        """
        Executes a controlled repository update in Gitea (writes markdown task entry and appends commit log).
        """
        timestamp = datetime.now().isoformat()
        commit_id = f"commit-{hash(timestamp + task_name) & 0xffffffff:08x}"

        # 1. Update TASKS.md
        entry = (
            f"\n## Task: {task_name}\n"
            f"- **Priority**: {priority} (Confidence: {probability * 100:.1f}%)\n"
            f"- **Updated At**: {timestamp}\n"
            f"- **Commit ID**: {commit_id}\n"
            f"- **Explanation**: {explanation}\n"
        )
        
        with open(self.tasks_file, "a", encoding="utf-8") as f:
            f.write(entry)

        # 2. Update Commit Log
        logs = []
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                try:
                    logs = json.load(f)
                except Exception:
                    logs = []

        commit_record = {
            "commit_id": commit_id,
            "timestamp": timestamp,
            "task_name": task_name,
            "priority": priority,
            "probability": probability,
            "action": "ADD_SCHEDULED_TASK"
        }
        logs.append(commit_record)

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)

        return {
            "status": "success",
            "commit_id": commit_id,
            "repo_file": str(self.tasks_file),
            "timestamp": timestamp
        }

    def verify_gitea_action(self, commit_id: str) -> Tuple[bool, str]:
        """
        INDEPENDENT VERIFICATION FUNCTION.
        Does NOT rely on the agent's self-report.
        Queries the repository log and file system directly to confirm the change exists.
        """
        if not self.tasks_file.exists() or not self.log_file.exists():
            return False, "Independent Verification Failed: Repository file or commit log does not exist."

        with open(self.log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)

        matching_commit = next((c for c in logs if c["commit_id"] == commit_id), None)
        if not matching_commit:
            return False, f"Independent Verification Failed: Commit ID '{commit_id}' not found in Gitea log."

        with open(self.tasks_file, "r", encoding="utf-8") as f:
            tasks_content = f.read()

        if commit_id not in tasks_content:
            return False, f"Independent Verification Failed: Commit ID '{commit_id}' missing from TASKS.md content."

        return True, f"Independent Verification Passed: Commit ID '{commit_id}' confirmed present in Gitea repository."
