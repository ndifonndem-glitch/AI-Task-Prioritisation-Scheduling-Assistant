"""
Generate a 1000-row synthetic dataset matching Dataset 2 schema.

Columns:
    snapshot_id, task_id, hours_to_deadline, est_duration_h,
    stated_importance, slack_ratio, fits_largest_block, dep_blocked,
    n_dependents, times_deferred, age_days,
    source_calendar, source_email, source_manual, source_message,
    source, label

Target: label = Do now / Schedule this week / Defer
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TOTAL_ROWS = 1000
SNAP_MIN, SNAP_MAX = 5, 10
LARGEST_BLOCK_H = 8.0      # hours available in the largest free block

TASK_NAMES = [
    "Prepare for the board meeting",
    "Complete the onboarding paperwork",
    "Analyse the operational data and prepare insights",
    "Set up monitoring alerts on staging",
    "Conduct the screening interview with the candidate",
    "Complete the server migration",
    "Complete the final architecture document",
    "Conduct data analysis on data",
    "Respond to HR emails about the budget",
    "Complete the preliminary project proposal document",
    "Write the agenda for the all-hands",
    "Submit the compliance report",
    "Complete the quarterly project proposal document",
    "Research best practices and summarise findings",
    "Conduct risk analysis on risk",
    "Complete the timesheet",
    "Respond to the CEO emails about the budget",
    "Prepare the sample dataset for the client demo",
    "Prepare for the investor meeting",
    "Conduct the panel interview with the candidate",
    "Deploy the update to staging",
    "Follow up with the client about the feedback",
    "Update the team wiki",
    "Investigate and resolve the authentication issue",
    "Deliver the final report to the partner",
    "Complete the testing phase of the mobile project",
    "Analyse the log data and prepare insights",
    "Refactor the payment module",
    "Prepare the live demo for the client demo",
    "Research the competition and summarise findings",
    "Organise and archive meeting recordings",
    "Prepare the prototype walkthrough for the client demo",
    "Follow up with the stakeholder about the next steps",
    "Organise and archive the shared drive",
    "Send the meeting summary to the development team",
    "Complete the database backup",
    "Write unit tests for the data pipeline",
    "Complete the design phase of the AI project",
    "Write the job description",
    "Submit the financial summary",
    "Complete the security patch",
    "Deploy the hotfix to staging",
    "Review the backend pull request",
    "Complete the capacity review",
]

DEPENDENCY_NOTES = [
    "the client feedback",
    "the design review",
    "the data analysis",
    "code review",
    "the security audit",
    "the legal review",
    "the database migration",
    "stakeholder sign-off",
    "management approval",
    "QA testing",
    "peer review",
    "the prototype",
    "the vendor response",
    "the requirements doc",
    "the kickoff meeting",
]

SOURCES = ["calendar", "email", "manual", "message"]
SOURCE_PROBS = [0.30, 0.30, 0.20, 0.20]


# ----------------------------------------------------------------------
# Feature sampling
# ----------------------------------------------------------------------
def sample_deadline():
    """Sample hours_to_deadline from a realistic mix of buckets."""
    bucket = np.random.choice(["urgent", "soon", "later", "eventual"],
                              p=[0.30, 0.35, 0.25, 0.10])
    if bucket == "urgent":
        return float(np.random.uniform(1, 12))
    if bucket == "soon":
        return float(np.random.uniform(12, 72))
    if bucket == "later":
        return float(np.random.uniform(72, 200))
    return float(np.random.uniform(200, 340))


def sample_duration():
    """Sample est_duration_h from short / medium / long buckets."""
    bucket = np.random.choice(["short", "medium", "long"], p=[0.45, 0.40, 0.15])
    if bucket == "short":
        return float(np.random.uniform(0.25, 2.0))
    if bucket == "medium":
        return float(np.random.uniform(2.0, 6.0))
    return float(np.random.uniform(6.0, 12.0))


def sample_importance():
    """Stated importance 1-5 with a reasonable distribution."""
    return int(np.random.choice([1, 2, 3, 4, 5],
                                p=[0.15, 0.20, 0.25, 0.20, 0.20]))


def sample_dependency():
    """Whether the task is blocked, and by what."""
    blocked = int(np.random.choice([0, 1], p=[0.78, 0.22]))
    note = str(np.random.choice(DEPENDENCY_NOTES)) if blocked else ""
    return blocked, note


def sample_system_fields():
    """Fields the LLM cannot extract: system state, history, source."""
    n_dependents = int(np.random.choice(
        [0, 1, 2, 3, 4, 5],
        p=[0.60, 0.20, 0.10, 0.05, 0.03, 0.02]
    ))
    
    probs = np.array([0.20, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05,
                      0.04, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.02])
    probs = probs / probs.sum()
    
    times_deferred = int(np.random.choice(
        range(0, 16),
        p=probs
    ))
    age_days = int(np.random.randint(0, 16))
    source = str(np.random.choice(SOURCES, p=SOURCE_PROBS))
    return n_dependents, times_deferred, age_days, source


# ----------------------------------------------------------------------
# Label rules (deterministic + noise)
# ----------------------------------------------------------------------
def assign_label(hours_to_deadline, est_duration_h, stated_importance,
                 dep_blocked, times_deferred, fits_largest_block,
                 slack_ratio):
    """
    Rule-based label with 5% noise.
    """
    if dep_blocked == 1:
        label = "Defer" if slack_ratio > 5 else "Schedule this week"
    elif (slack_ratio < 2 and stated_importance >= 4 and fits_largest_block == 1):
        label = "Do now"
    elif (slack_ratio < 8 and stated_importance >= 3 and times_deferred <= 5):
        label = "Schedule this week"
    else:
        label = "Defer"

    # 5% label noise
    if np.random.random() < 0.05:
        label = str(np.random.choice(["Do now", "Schedule this week", "Defer"]))

    return label


# ----------------------------------------------------------------------
# Main generator
# ----------------------------------------------------------------------
def generate_dataset(total_rows=TOTAL_ROWS):
    rows = []
    task_counter = 1
    snapshot_counter = 1

    while len(rows) < total_rows:
        snap_size = int(np.random.randint(SNAP_MIN, SNAP_MAX + 1))
        if len(rows) + snap_size > total_rows:
            snap_size = total_rows - len(rows)

        snapshot_id = f"S{snapshot_counter:03d}"

        for _ in range(snap_size):
            task_id = f"{snapshot_id}-T{task_counter:02d}"
            task_counter += 1

            hours_to_deadline = sample_deadline()
            est_duration_h = sample_duration()
            stated_importance = sample_importance()
            dep_blocked, dependency_note = sample_dependency()
            n_dependents, times_deferred, age_days, source = sample_system_fields()

            # Derived features
            slack_ratio = (hours_to_deadline - est_duration_h) / max(est_duration_h, 0.5)
            slack_ratio = max(slack_ratio, 0.01)
            fits_largest_block = int(est_duration_h <= LARGEST_BLOCK_H)

            label = assign_label(
                hours_to_deadline, est_duration_h, stated_importance,
                dep_blocked, times_deferred, fits_largest_block, slack_ratio
            )

            rows.append({
                "snapshot_id": snapshot_id,
                "task_id": task_id,
                "task_name": np.random.choice(TASK_NAMES),
                "hours_to_deadline": round(hours_to_deadline, 2),
                "est_duration_h": round(est_duration_h, 2),
                "stated_importance": stated_importance,
                "slack_ratio": round(slack_ratio, 3),
                "fits_largest_block": fits_largest_block,
                "dep_blocked": dep_blocked,
                "dependency_note": dependency_note,
                "n_dependents": n_dependents,
                "times_deferred": times_deferred,
                "age_days": age_days,
                "source_calendar": int(source == "calendar"),
                "source_email": int(source == "email"),
                "source_manual": int(source == "manual"),
                "source_message": int(source == "message"),
                "source": source,
                "label": label,
            })

        snapshot_counter += 1

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = generate_dataset(1000)
    
    # Save to data directory
    primary_file = output_dir / "knowledge_worker_task_priority_5000.csv"
    secondary_file = output_dir / "dataset2_1000.csv"
    
    df.to_csv(primary_file, index=False)
    df.to_csv(secondary_file, index=False)

    print("=" * 60)
    print("NEW ML DATASET GENERATED (Dataset 2 Schema)")
    print("=" * 60)
    print("Rows:", len(df))
    print("Snapshots:", df["snapshot_id"].nunique())
    print(f"Saved to: {primary_file}")
    print("\nLabel distribution:")
    print(df["label"].value_counts(normalize=True).mul(100).round(2))
    print("\nFirst 5 rows:")
    print(df.head())