#!/usr/bin/env python3
"""Task 10 — an end-to-end harness that actually goes end to end.

The existing harness passes structured parameters straight into the agent, so it never
exercises the language model. A failure in extraction could not be detected by it, and
it reports 10/10 because the cases were chosen to cover the three outcomes rather than
to be hard.

This drives the OpenClaw gateway with unstructured sentences, fixes the pass criteria
in source before any run, repeats each case, and attributes every failure to a named
component.

    python3 7_test_e2e_natural_language.py --runs 3
"""
import argparse, json, re, statistics, subprocess, sys, time

# ---------------------------------------------------------------- pass criteria
# Fixed here, in advance, so they cannot be adjusted after seeing the results.
CRITERIA = {
    "extraction_fields_required": ["hours_to_deadline", "est_duration_h", "stated_importance"],
    "priority_must_be_one_of": ["Do now", "Schedule this week", "Defer"],
    "max_seconds_per_run": 900,
    "gitea_must_verify": True,
    "min_pass_rate_to_claim_working": 0.80,
}

# Deliberately includes cases near the tree's split points and ambiguous phrasing.
CASES = [
    dict(id="clear-urgent", expect="Do now",
         text="The client demo deck is due in 3 hours, it needs about 2 hours of work, "
              "I have 4 hours free, and it is the most important thing today."),
    dict(id="clear-defer", expect="Defer",
         text="At some point I should tidy up the old project archive. No deadline, "
              "maybe half a day of work, not important."),
    dict(id="boundary-slack", expect=None,          # near slack_ratio 7.254
         text="The compliance summary is due in about 15 hours and will take roughly "
              "two hours. I have most of tomorrow free. Fairly important."),
    dict(id="boundary-importance", expect=None,     # near stated_importance 2.5
         text="Someone asked me to review the onboarding doc this week. It is a couple "
              "of hours of work. Middling priority, I suppose."),
    dict(id="ambiguous-deadline", expect=None,
         text="Need to get the migration notes done before the team meets - that is "
              "either Thursday or Friday, I think. Half a day of work. Quite important."),
    dict(id="blocked-dependency", expect=None,
         text="I cannot start the integration test until the vendor sends credentials. "
              "Due in two days, about three hours of work, important."),
]


def run_agent(text, session, timeout):
    """One turn through the real gateway. No shortcuts into the agent's internals."""
    cmd = ["openclaw", "agent", "--agent", "main",
           "--session-key", session, "-m", text, "--json"]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"ok": p.returncode == 0, "stdout": p.stdout, "stderr": p.stderr,
                "seconds": round(time.time() - t0, 1)}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "timeout",
                "seconds": round(time.time() - t0, 1)}
    except FileNotFoundError:
        return {"ok": False, "stdout": "", "stderr": "no-gateway",
                "seconds": round(time.time() - t0, 1)}


def attribute(result, gitea_state):
    """Name the component responsible. 'Something failed' is not an evaluation."""
    out = (result.get("stdout") or "") + (result.get("stderr") or "")
    if result.get("stderr") == "no-gateway":
        return "infrastructure", "the openclaw CLI is not on PATH - run this on CT 201"
    if result.get("stderr") == "timeout":
        return "infrastructure", "run exceeded the time budget before completing"
    if "FAILED_API_GUARDRAIL" in out:
        return "prediction-service", "the agent reported the service unreachable"
    if not re.search(r"predict", out, re.I):
        return "agent", "no evidence the prediction tool was called"
    if not any(p in out for p in CRITERIA["priority_must_be_one_of"]):
        return "llm-extraction", "no valid priority appears in the agent's reply"
    if CRITERIA["gitea_must_verify"] and not gitea_state.get("verified"):
        if gitea_state.get("mode") == "MOCK":
            return "gitea", "the write went to the mock repository, not the server"
        return "gitea", "the write could not be verified by an independent read"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=CRITERIA["max_seconds_per_run"])
    args = ap.parse_args()

    print("Pass criteria, fixed in source before this run:")
    print(json.dumps(CRITERIA, indent=2), "\n")

    results, failures = [], []
    for run in range(1, args.runs + 1):
        for case in CASES:
            session = f"e2e-{case['id']}-r{run}-{int(time.time())}"
            r = run_agent(case["text"], session, args.timeout)
            # verification state is read from the agent's own result payload; the
            # tool now reports mode and verified explicitly (see 5_gitea_no_mock.py)
            gitea = {}
            m = re.search(r'"mode"\s*:\s*"(\w+)"', r["stdout"] or "")
            v = re.search(r'"verified"\s*:\s*(true|false)', r["stdout"] or "")
            if m: gitea["mode"] = m.group(1)
            if v: gitea["verified"] = v.group(1) == "true"
            component, why = attribute(r, gitea)
            passed = component is None
            if case["expect"] and passed:
                if case["expect"] not in (r["stdout"] or ""):
                    passed, component, why = False, "decision-tree", \
                        f"expected {case['expect']} for an unambiguous case"
            results.append(dict(run=run, case=case["id"], passed=passed,
                                seconds=r["seconds"], component=component, why=why))
            if not passed:
                failures.append(results[-1])
            print(f"  run {run}  {case['id']:<22} "
                  f"{'pass' if passed else 'FAIL'}  {r['seconds']:>6.1f}s"
                  + ("" if passed else f"  -> {component}: {why}"))

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    rate = passed / total if total else 0
    print(f"\n{passed}/{total} passed ({rate:.0%})")
    if results:
        print(f"median turn: {statistics.median(r['seconds'] for r in results):.1f}s")

    if failures:
        print("\nFailures by component:")
        by = {}
        for f in failures:
            by.setdefault(f["component"], []).append(f["case"])
        for comp, cases in sorted(by.items(), key=lambda kv: -len(kv[1])):
            print(f"  {comp:<20} {len(cases):>2}  ({', '.join(sorted(set(cases)))})")

    json.dump({"criteria": CRITERIA, "results": results},
              open("e2e_natural_language_results.json", "w"), indent=1)
    print("\nwrote e2e_natural_language_results.json")
    print("Report the rate this produces, including a rate below 100%. A harness that "
          "never fails cannot attribute failures, which is the criterion.")
    return 0 if rate >= CRITERIA["min_pass_rate_to_claim_working"] else 1


if __name__ == "__main__":
    sys.exit(main())
