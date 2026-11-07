"""Task 10 — base against fine-tuned through the router, model name the only variable.

Measures three things separately, because they are not the same thing:
  schema_ok  - valid JSON with the required keys
  bare_json  - emitted without markdown fences (what the fine-tune was meant to teach)
  numeric_ok - the extracted numbers match what the sentence actually says
"""
import json, re, time, urllib.request

ROUTER = "http://127.0.0.1:8080/v1/chat/completions"
INSTR = ("Extract the task parameters as JSON with keys task, hours_to_deadline, "
         "est_duration_h, importance. Reply with JSON only.\nTask: ")

CASES = [
    ("demo deck",  "finish the client demo deck, due in 3 hours, about 2 hours of work, very important", 3, 2),
    ("audit",      "review the Q4 security audit, due in 30 hours, roughly 6 hours of work, fairly important", 30, 6),
    ("migration",  "write the migration notes, needed in 12 hours, 4 hours of work, important", 12, 4),
    ("archive",    "tidy the old project archive, due in 200 hours, 5 hours of work, not important", 200, 5),
    ("onboarding", "review the onboarding doc, due in 48 hours, 2 hours of work, medium priority", 48, 2),
    ("patch",      "deploy the security patch, due in 1 hour, half an hour of work, critical", 1, 0.5),
]

def ask(model, text):
    payload = {"model": model, "temperature": 0, "max_tokens": 120,
               "messages": [{"role": "user", "content": INSTR + text}]}
    req = urllib.request.Request(ROUTER, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"].get("content", ""), time.time() - t0

def grade(text, hrs, dur):
    bare = not text.strip().startswith("```")
    try:
        o = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except Exception:
        return False, bare, False
    schema = all(k in o for k in ("task", "hours_to_deadline", "est_duration_h", "importance"))
    try:
        numeric = (abs(float(o["hours_to_deadline"]) - hrs) < 0.01
                   and abs(float(o["est_duration_h"]) - dur) < 0.01)
    except Exception:
        numeric = False
    return schema, bare, numeric

for model in ("qwen05b-base", "qwen05b-taskextract"):
    s = b = n = 0
    secs = []
    for name, text, hrs, dur in CASES:
        try:
            out, t = ask(model, text)
            sc, ba, nu = grade(out, hrs, dur)
            s += sc; b += ba; n += nu; secs.append(t)
            print("  %-18s %-11s schema=%-5s bare=%-5s numeric=%-5s %4.1fs"
                  % (model, name, sc, ba, nu, t))
        except Exception as e:
            print("  %-18s %-11s ERROR %s" % (model, name, e))
    k = len(CASES)
    print("  -> %-18s schema %d/%d  bare JSON %d/%d  numeric %d/%d  mean %.1fs\n"
          % (model, s, k, b, k, n, k, sum(secs) / max(len(secs), 1)))
