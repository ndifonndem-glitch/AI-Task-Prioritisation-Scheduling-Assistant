# IT8532 Assessment 2: Technical Architecture Documentation
**System:** AI-Powered Knowledge Worker Task Prioritisation & Scheduling Assistant  
**Target Infrastructure:** Proxmox CT 200 (`inference-llm`, IP: `192.168.10.20`, 8 GB RAM, CPU-only) + CT 201 + Gitea

---

## 1. System Overview & Component Division

The system addresses the knowledge worker scheduling problem by assigning distinct tasks to the AI paradigm best suited to solve them:

```
┌────────────────────────────────────────────────────────┐
│                      USER INPUT                        │
└──────────────────────────┬─────────────────────────────┘
                           │ (Natural Language Task)
                           ▼
┌────────────────────────────────────────────────────────┐
│                    OPENCLAW AGENT                      │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
               ▼                          ▼
┌────────────────────────┐     ┌─────────────────────────┐
│    FINE-TUNED LLM      │     │    CONVENTIONAL ML      │
│   (Qwen2.5-3B GGUF)    │     │(Decision Tree Classifier)│
└──────────────┬─────────┘     └──────────┬──────────────┘
               │                          │
               │                          ▼
               │               ┌─────────────────────────┐
               │               │     PREDICTION API      │
               │               └──────────┬──────────────┘
               │                          │
               └──────────┬───────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│                    AGENT DECISION                      │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│                       GUARDRAILS                       │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
               ▼                          ▼
┌────────────────────────┐     ┌─────────────────────────┐
│    HUMAN APPROVAL      │     │    LOW-RISK ACTION      │
└──────────────┬─────────┘     └──────────┬──────────────┘
               │                          │
               └──────────┬───────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│              GITEA REPOSITORY INTEGRATION              │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│             INDEPENDENTLY VERIFIED RESULT              │
└────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Component A: Conventional ML Priority Classifier (Symbolist Tribe)
* **Algorithm:** `DecisionTreeClassifier` (Scikit-Learn).
* **Rationale:** Structured tabular variables naturally form explicit decision rules (e.g. `IF deadline_hours < 12 AND importance >= 4 THEN HIGH`). Provides complete interpretability, deterministic output, and low CPU footprint.
* **Service Layer:** FastAPI REST server (`POST /predict`) on port 5000 with Pydantic request validation and HTTP 400 error handling for malformed input.

### 2.2 Component B: Fine-Tuned Language Model (Foundationist Paradigm)
* **Base Model:** `Qwen2.5-3B-Instruct`
* **Fine-Tuning:** Unsloth QLoRA (4-bit quantization, rank=16, alpha=32) trained on 1,000 instruction examples.
* **Deployment Format:** Merged GGUF `Q4_K_M` quantization served via CPU inference router (`llama.cpp` server) on port 8080.
* **Role:** Natural language understanding, JSON parameter extraction, and human-readable explanation generation. Does NOT make numerical priority decisions.

### 2.3 Component C: OpenClaw Agent Orchestration & Gitea Integration
* **Role:** Coordinates language extraction, calls Component A REST API tool, evaluates guardrails, and executes authorized repository updates.
* **Stopping Condition:** Immediately terminates upon Gitea action execution and independent verification confirmation.
* **Independent Verification:** Direct API/filesystem query verifying commit ID presence in `TASKS.md` and `commit_log.json` without relying on agent self-reports.

---

## 3. Safety Guardrails & Human Approval Boundary

1. **Failure Mode 1 (Invalid Input):** Strict schema validation rejects negative durations or out-of-bounds importance ratings prior to model invocation.
2. **Failure Mode 2 (API Failure Fallback):** If prediction REST API is unreachable, agent logs error and requests user retry; no fabricated fallback predictions are generated.
3. **Failure Mode 3 (Human Approval Boundary):** Any action attempting to delete/displace a HIGH priority task or modify protected branches (`main`/`master`) pauses agent execution and triggers a required human confirmation step.

---

## 4. Empirical Evaluation Summary

* **ML Model F1-Score:** `0.8836` (beats Majority Dummy `0.4862` and Logistic Regression `0.8353`).
* **LLM Benchmark:** Fine-tuned GGUF model achieves `70.0%` strict JSON schema compliance vs `0.0%` for Base LLM.
* **End-to-End System Benchmark:** 10/10 (100.0%) PASS rate across repeated multi-scenario test suite.
