# llm/export

GGUF artefacts served by the inference backends. The files are not committed — they are
hundreds of megabytes each — so this directory holds only the manifest below. The
deployment mounts it read-only at `/models` (see `deployment/docker-compose.yml`).

| File | Role | Size (bytes) | sha256 prefix | Port |
|---|---|---|---|---|
| `qwen2.5-0.5b-instruct-base.Q4_K_M.gguf` | base, control arm | 491,400,032 | `74a4da8c9fdbcd15` | 8082 |
| `qwen05b-taskextract.Q4_K_M.gguf` | fine-tuned | 397,807,584 | `51e49f8cd902f820` | 8081 |
| `qwen2.5-3b-instruct-q4_k_m.gguf` | 3B agent reasoning model | 1,929,902,592 | — | 8083 |

**The base and fine-tuned files must differ.** `deployment/separate_models.sh` provisions
them and fails if the two hash identically; `tests/test_router_registration.py` checks
both the configuration and the running endpoints.

To populate this directory, run `deployment/separate_models.sh` on the inference host.
