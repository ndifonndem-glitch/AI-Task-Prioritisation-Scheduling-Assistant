"""
Generate an instruction dataset for fine-tuning a small LLM.

Format: JSONL, one example per line.
    {"instruction": ..., "input": ..., "output": ...}

The LLM learns to extract structured scheduling features from
unstructured task text. The output schema matches the features
the ML model consumes.

Output schema (always the same keys, same types):
    {
      "task": str,
      "hours_to_deadline": float,
      "est_duration_h": float,
      "urgency": int 1-5,
      "importance": int 1-5,
      "has_dependency": 0 or 1,
      "dependency_note": str
    }
"""

import json
import random
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)

# ======================================================================
# Configuration
# ======================================================================
N_EXAMPLES = 1200
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "llm_instruction_data.jsonl"
TRAIN_FILE = OUTPUT_DIR / "llm_train.jsonl"
VAL_FILE = OUTPUT_DIR / "llm_val.jsonl"
TEST_FILE = OUTPUT_DIR / "llm_test.jsonl"
FINETUNE_JSON = OUTPUT_DIR / "finetune_dataset.json"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


# ======================================================================
# Vocabulary
# ======================================================================
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
    "Conduct the technical interview with the candidate",
    "Fix the authentication bug in authentication",
    "Fix the database bug in database",
    "Investigate and resolve the performance issue on the server",
    "Investigate and resolve the storage issue on the API",
    "Set up load balancer on production",
    "Set up VPN access on UAT",
    "Set up CI/CD pipeline for the project",
    "Complete the SSL certificate renewal",
    "Complete the disaster recovery test",
    "Complete the training plan for the intern",
    "Complete the performance review for the new hire",
    "Write the employee handbook update",
    "Write the policy document",
    "Write the incident report",
    "Prepare the analysis for the board",
    "Prepare the summary for the team",
    "Prepare the briefing for the finance team",
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
    "the API integration",
    "the onboarding session",
    "the server setup",
    "the budget approval",
    "the initial research",
]

PREFIXES = [
    "", "", "", "", "",
    "Task: ", "TODO: ", "Reminder: ", "Remind me: ",
    "URGENT: ", "Emergency: ", "Escalation — need to ",
    "Background task: ", "Optional: ", "Non-urgent: ",
    "Flag: ", "Drop everything — ", "Critical — ",
    "This is important: ", "Please add: ",
    "Add to my list: ", "When capacity allows: ",
    "Can you remind me to ", "I need to ", "I should ",
    "Get ", "Need ", "Schedule ",
]

# Deadline phrases mapped to approximate hours
DEADLINE_PHRASES = [
    (["right now", "immediately", "by tonight", "ASAP"], 1.0),
    (["in 1 hour", "in 3 hours", "within the next 2 hours"], 4.0),
    (["by lunchtime today", "this afternoon",
      "by close of business today", "by end of today"], 8.0),
    (["by tomorrow morning", "by tomorrow afternoon"], 18.0),
    (["by tomorrow", "within 24 hours", "by end of tomorrow"], 24.0),
    (["by Wednesday", "within three days", "in a couple of days"], 48.0),
    (["by Thursday", "within five days"], 72.0),
    (["by Friday", "this week", "before the weekend",
      "by next Monday", "within a week"], 168.0),
    (["within two weeks", "by end of next week",
      "sometime next week"], 240.0),
    (["by the end of the month", "within two weeks"], 336.0),
    (["eventually", "when you get a chance",
      "no rush", "at some point this month"], 240.0),
]

# Duration phrases mapped to approximate hours
DURATION_PHRASES = [
    (["about 15 minutes", "roughly 15 minutes"], 0.25),
    (["about half an hour", "roughly 30 minutes"], 0.5),
    (["about 45 minutes"], 0.75),
    (["about an hour", "roughly an hour",
      "an hour or so", "about an hour or so"], 1.0),
    (["about 90 minutes", "around an hour and a half"], 1.5),
    (["a couple of hours", "about 2 hours", "roughly 2 hours"], 2.0),
    (["about 3 hours", "roughly 3 hours"], 3.0),
    (["about half a day", "around 4 hours"], 4.0),
    (["about 5 hours", "roughly 5 hours"], 5.0),
    (["about 6 hours", "most of the day"], 6.0),
    (["a full day", "about 8 hours"], 8.0),
    (["about 10 hours"], 10.0),
    (["a day and a half", "about 12 hours"], 12.0),
]

# Urgency markers -> 1-5
URGENCY_MARKERS = {
    5: ["URGENT", "immediately", "ASAP", "right now", "by tonight",
        "Emergency", "Drop everything", "within the next 2 hours",
        "by lunchtime today", "this afternoon",
        "by close of business today", "needed by end of today"],
    4: ["by end of today", "by tomorrow", "within 24 hours",
        "by tomorrow morning", "by tomorrow afternoon"],
    3: ["by Wednesday", "by Thursday", "within three days",
        "within five days", "in a couple of days"],
    2: ["by Friday", "this week", "before the weekend",
        "by next Monday", "within a week"],
    1: ["eventually", "when you get a chance", "no rush",
        "by the end of the month", "sometime next week",
        "at some point this month"],
}

# Importance markers -> 1-5
IMPORTANCE_MARKERS = {
    5: ["critical", "extremely important", "must not be missed",
        "mission critical", "absolutely essential",
        "very high stakes", "C-suite visibility"],
    4: ["very important", "fairly important", "key deliverable",
        "high priority", "important", "client-facing"],
    3: ["moderately important", "normal priority",
        "reasonably important", "standard priority",
        "should be done properly"],
    2: ["minor but needed", "not a high priority",
        "low-medium priority", "slightly important",
        "would be nice to finish"],
    1: ["not critical at all", "not important", "optional really",
        "low priority", "trivial", "can wait", "background task",
        "filler work"],
}


# ======================================================================
# Sampling helpers
# ======================================================================
def sample_deadline():
    phrases, hours = random.choice(DEADLINE_PHRASES)
    phrase = random.choice(phrases)
    jitter = float(np.random.uniform(0.9, 1.1))
    return phrase, round(hours * jitter, 2)


def sample_duration():
    phrases, hours = random.choice(DURATION_PHRASES)
    phrase = random.choice(phrases)
    jitter = float(np.random.uniform(0.9, 1.1))
    return phrase, round(hours * jitter, 2)


def sample_urgency():
    level = int(np.random.choice([1, 2, 3, 4, 5],
                                 p=[0.20, 0.20, 0.20, 0.20, 0.20]))
    return level, random.choice(URGENCY_MARKERS[level])


def sample_importance():
    level = int(np.random.choice([1, 2, 3, 4, 5],
                                 p=[0.15, 0.20, 0.25, 0.20, 0.20]))
    return level, random.choice(IMPORTANCE_MARKERS[level])


def sample_dependency():
    blocked = int(np.random.choice([0, 1], p=[0.75, 0.25]))
    if not blocked:
        templates = [
            "It does not depend on another task.",
            "No dependencies.",
            "Not blocked by anything.",
            "",
            "",
        ]
        return 0, "", random.choice(templates)

    note = random.choice(DEPENDENCY_NOTES)
    templates = [
        f"Blocked — depends on {note} being finished first.",
        f"Only possible after {note}.",
        f"Cannot start until {note} is approved.",
        f"Waiting on {note} before I can begin.",
        f"Contingent on {note}.",
        f"It is blocked by {note}.",
        f"Note — it is blocked by {note}.",
        f"However, it is need {note} to be completed first.",
        f"This task is depends on {note} being finished first.",
        f"It is waiting on {note} before I can begin.",
    ]
    return 1, note, random.choice(templates)


# ======================================================================
# Compose natural language input
# ======================================================================
def compose_input(task_name, prefix, deadline_phrase, duration_phrase,
                  urgency_marker, importance_marker, dependency_sentence):
    """
    Build a natural-sounding task description from parts.
    """
    template = random.choice(["A", "B", "C", "D", "E"])

    if template == "A":
        parts = []
        if prefix:
            parts.append(prefix.rstrip(" —"))
        parts.append(f"{task_name} {deadline_phrase}.")
        parts.append(f"Takes {duration_phrase}.")
        if importance_marker:
            parts.append(f"It is {importance_marker}.")
        if dependency_sentence:
            parts.append(dependency_sentence)
        text = " ".join(parts)

    elif template == "B":
        head = f"{prefix}{task_name}" if prefix else task_name
        text = f"{head} — {deadline_phrase}. "
        text += f"Takes about {duration_phrase}. "
        if importance_marker:
            text += f"It is {importance_marker}. "
        text += dependency_sentence

    elif template == "C":
        intro = prefix if prefix else random.choice(
            ["I need to ", "I should ", "Please ", "Can you remind me to "]
        )
        text = f"{intro}{task_name.lower()} {deadline_phrase}. "
        text += f"I think it will take {duration_phrase}. "
        if importance_marker:
            text += f"It is {importance_marker}. "
        text += dependency_sentence

    elif template == "D":
        head = prefix if prefix else "Task: "
        text = f"{head}{task_name} — {deadline_phrase}. "
        if dependency_sentence:
            text += f"Note: {dependency_sentence} "
        text += f"Duration: {duration_phrase}."
        if importance_marker:
            text += f" Priority: {importance_marker}."

    else:
        text = f"Get {task_name} done {deadline_phrase}. "
        text += f"About {duration_phrase} of work. "
        if importance_marker:
            text += f"This is {importance_marker}. "
        text += dependency_sentence

    text = " ".join(text.split())
    text = text.replace(" .", ".").replace("..", ".")
    text = text.replace(" ,", ",").replace(" ;", ";")
    return text.strip()


# ======================================================================
# Build output JSON
# ======================================================================
def build_output(task_name, hours_to_deadline, est_duration_h,
                 urgency, importance, has_dependency, dependency_note):
    obj = {
        "task": task_name,
        "hours_to_deadline": float(hours_to_deadline),
        "est_duration_h": float(est_duration_h),
        "urgency": int(urgency),
        "importance": int(importance),
        "has_dependency": int(has_dependency),
        "dependency_note": dependency_note if has_dependency else "",
    }
    return json.dumps(obj, ensure_ascii=False)


# ======================================================================
# Main generator
# ======================================================================
def generate_examples(n):
    examples = []
    seen = set()

    while len(examples) < n:
        task_name = random.choice(TASK_NAMES)
        prefix = random.choice(PREFIXES)

        deadline_phrase, hours_to_deadline = sample_deadline()
        duration_phrase, est_duration_h = sample_duration()
        urgency, urgency_marker = sample_urgency()
        importance, importance_marker = sample_importance()
        has_dependency, dependency_note, dependency_sentence = sample_dependency()

        if prefix in ("URGENT: ", "Emergency: ", "Drop everything — "):
            urgency = 5
        if prefix in ("Non-urgent: ", "Optional: ", "Background task: "):
            urgency = 1

        if deadline_phrase in ("right now", "immediately", "by tonight", "ASAP"):
            if prefix in ("Non-urgent: ", "Optional: ", "Background task: "):
                prefix = ""

        text = compose_input(
            task_name, prefix, deadline_phrase, duration_phrase,
            urgency_marker, importance_marker, dependency_sentence
        )

        key = text.lower()
        if key in seen or len(text) < 20:
            continue
        seen.add(key)

        output = build_output(
            task_name, hours_to_deadline, est_duration_h,
            urgency, importance, has_dependency, dependency_note
        )

        example = {
            "instruction": "Extract structured scheduling features from the task description.",
            "input": text,
            "output": output,
        }
        examples.append(example)

    return examples


def save_jsonl(records, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_sharegpt_json(records, filepath):
    system_prompt = (
        "You are a task extraction assistant for a time management system. "
        "Extract scheduling information from the user's natural language task "
        "description and return ONLY a valid JSON object with exactly these fields: "
        "task, hours_to_deadline, est_duration_h, urgency, importance, has_dependency, dependency_note."
    )
    sharegpt_list = []
    for r in records:
        sharegpt_list.append({
            "conversations": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": r["input"]},
                {"role": "assistant", "content": r["output"]}
            ]
        })
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sharegpt_list, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("=" * 60)
    print("LLM INSTRUCTION DATASET GENERATOR")
    print("=" * 60)

    data = generate_examples(N_EXAMPLES)

    # Train / Val / Test split
    train, test_val = train_test_split(data, test_size=(VAL_RATIO + TEST_RATIO), random_state=42)
    val, test = train_test_split(test_val, test_size=0.50, random_state=42)

    save_jsonl(data, OUTPUT_FILE)
    save_jsonl(train, TRAIN_FILE)
    save_jsonl(val, VAL_FILE)
    save_jsonl(test, TEST_FILE)
    
    # Also save single unified finetune_dataset.json file for Unsloth
    save_sharegpt_json(data, FINETUNE_JSON)

    print(f"Total examples : {len(data)}")
    print(f"Train split    : {len(train)} -> {TRAIN_FILE}")
    print(f"Val split      : {len(val)} -> {VAL_FILE}")
    print(f"Test split     : {len(test)} -> {TEST_FILE}")
    print(f"Full JSONL     : {OUTPUT_FILE}")
    print(f"Full ShareGPT  : {FINETUNE_JSON}")
    print("\nSample instruction example:")
    print(json.dumps(data[0], indent=2))
    print("\nDataset generation complete.")