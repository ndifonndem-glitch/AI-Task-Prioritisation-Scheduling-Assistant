# IT8532 Assessment 2: AI Task Prioritisation & Scheduling Assistant

An end-to-end 3-component AI subsystem built for Layer 2 of the agentic enterprise stack, targeting deployment on **Proxmox CT 200** (`inference-llm`, IP: `192.168.10.20`, 8 GB RAM, CPU-only).

---

## 🏛️ System Architecture

```
User Task Description -> Fine-Tuned LLM (JSON Extraction) -> Decision Tree REST API (Priority Score) -> Guardrails -> OpenClaw Agent -> Gitea Commit -> Independent Verification
```

* **Component A (Prediction Service):** Scikit-Learn Decision Tree (`POST /predict` on FastAPI).
* **Component B (Language Model):** `Qwen2.5-3B-Instruct` fine-tuned with Unsloth QLoRA, exported to GGUF (`Q4_K_M`).
* **Component C (Agent Orchestration):** OpenClaw Agent calling REST tool, enforcing guardrails, and logging verified Gitea commits.

---

## 🚀 Quickstart & Execution

### 1. Data Generation
```powershell
py generate_ml_dataset.py
py generate_llm_dataset.py
```

### 2. Train ML Model & Export Artifacts
```powershell
py notebooks/model_training.py
```

### 3. Run FastAPI REST API Server
```powershell
py -m uvicorn ml_service.main:app --host 0.0.0.0 --port 5000
```

### 4. Run Pytest Suite & Guardrail Verification
```powershell
py -m pytest tests/test_api.py -v
py -m pytest tests/test_guardrails.py -v
```

### 5. Run Base vs Fine-Tuned LLM Comparison Benchmark
```powershell
py llm/evaluation/compare_models.py
```

### 6. Run 10-Run End-to-End System Evaluation Harness
```powershell
py tests/test_e2e.py
```

---

## 📊 Verification Metrics

* **ML Model F1-Score:** `0.8836` (vs Majority Class Baseline `0.4862`)
* **API Test Suite:** 7/7 PASSED
* **Guardrail Test Suite:** 6/6 PASSED
* **End-to-End System Benchmark:** 10/10 PASSED (100.0%)

---

## 📄 License & Documentation
* [Architecture Documentation](docs/architecture.md)
* [Implementation Plan](docs/implementation_plan.md)
