"""Task 7 — both models registered, and demonstrably two different models.

The earlier version of this test asserted on strings in docker-compose.yml. That cannot
detect the defect it was meant to catch: when both services pointed at the same GGUF,
every string assertion still passed. Configuration proves intent; only the running
endpoints prove identity.

Three checks, in increasing strength:
  1. the two --model paths in docker-compose.yml differ
  2. the router lists both model names at /v1/models
  3. the two names return different output for the same prompt

Check 3 is the one that matters. Two names backed by one file pass 1 and 2 only if the
configuration is wrong in an obvious way; they cannot pass 3.

    python3 tests/test_router_registration.py            # all three
    python3 tests/test_router_registration.py --config   # check 1 only, no server needed
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROUTER = os.environ.get("LLM_ROUTER_URL", "http://127.0.0.1:8080").rstrip("/").removesuffix("/v1")
BASE_MODEL = os.environ.get("BASE_MODEL_NAME", "qwen05b-base")
FT_MODEL = os.environ.get("FT_MODEL_NAME", "qwen05b-taskextract")
COMPOSE = Path(__file__).resolve().parent.parent / "deployment" / "docker-compose.yml"


def check_config():
    assert COMPOSE.exists(), f"missing {COMPOSE}"
    text = COMPOSE.read_text(encoding="utf8")
    models = re.findall(r'"--model"\s*\n\s*-\s*"([^"]+)"', text)
    assert len(models) >= 2, f"expected at least two --model entries, found {models}"
    assert len(set(models)) == len(models), (
        f"the inference services point at the same GGUF, so 'base' and 'fine-tuned' are "
        f"one model under two names: {models}")
    print("1. config     OK  distinct model paths: %s" % [Path(m).name for m in models])
    return models


def _post(payload, timeout=180):
    req = urllib.request.Request(ROUTER + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def check_registered():
    with urllib.request.urlopen(ROUTER + "/v1/models", timeout=20) as r:
        listed = [m["id"] for m in json.loads(r.read().decode())["data"]]
    for name in (BASE_MODEL, FT_MODEL):
        assert name in listed, f"{name} not registered; router lists {listed}"
    print("2. registered OK  /v1/models lists: %s" % listed)


def check_distinct_behaviour():
    """Same prompt, same settings, only the model name differs."""
    prompt = ("Extract the task parameters as JSON with keys task, hours_to_deadline, "
              "est_duration_h, importance. Reply with JSON only.\n"
              "Task: finish the client demo deck, due in 3 hours, about 2 hours of work, "
              "very important.")
    payload = {"temperature": 0, "max_tokens": 120,
               "messages": [{"role": "user", "content": prompt}]}
    a = dict(payload, model=BASE_MODEL)
    b = dict(payload, model=FT_MODEL)
    differing = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
    assert differing == ["model"], f"payloads differ in more than the model name: {differing}"

    out_a = _post(a)["choices"][0]["message"].get("content", "")
    out_b = _post(b)["choices"][0]["message"].get("content", "")
    assert out_a.strip() and out_b.strip(), "one backend returned nothing"
    assert out_a.strip() != out_b.strip(), (
        "both model names produced byte-identical output for the same prompt at "
        "temperature 0. That is what one model served under two names looks like.")
    print("3. identity   OK  the two backends produce different output for one prompt")
    print("     base       : %s" % out_a.replace("\n", " ")[:88])
    print("     fine-tuned : %s" % out_b.replace("\n", " ")[:88])


def test_router_registration():
    check_config()
    if "--config" in sys.argv:
        print("\n(--config given: skipped the running-endpoint checks)")
        return
    check_registered()
    check_distinct_behaviour()


if __name__ == "__main__":
    try:
        test_router_registration()
        print("\nPASS")
    except AssertionError as e:
        print("\nFAIL: %s" % e); sys.exit(1)
    except Exception as e:
        print("\nERROR (%s): %s" % (type(e).__name__, e))
        print("The router must be running for checks 2 and 3; use --config for check 1 alone.")
        sys.exit(2)
