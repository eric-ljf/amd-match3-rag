#!/usr/bin/env python3
"""End-to-end CPU test: build sample corpus -> --index -> --query -> validate JSON.

Backend: tries EMB_BACKEND env, else sentence-transformers if importable,
else the deterministic hash backend (shape-only validation).
Exit 0 on pass, 1 on failure.
"""
import json
import os
import shutil
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTDIR = "/tmp/match3_test"
CORPUS = os.path.join(PROJ, "tests", "corpus_sample")
INDEX_DIR = os.path.join(TESTDIR, "index")
OUTPUT_DIR = os.path.join(TESTDIR, "output")

FAILURES = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def pick_backend():
    forced = os.environ.get("EMB_BACKEND")
    if forced:
        return forced
    try:
        import sentence_transformers  # noqa
        return "sbert"
    except Exception:
        print("sentence_transformers not importable; using hash backend (shape test only)")
        return "hash"


def run_app(args, extra_env=None):
    env = dict(os.environ)
    env["APP_INDEX_DIR"] = INDEX_DIR
    env["APP_OUTPUT_DIR"] = OUTPUT_DIR
    env["EMB_BACKEND"] = BACKEND
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, os.path.join(PROJ, "app.py")] + args,
        cwd=PROJ, env=env, capture_output=True, text=True, timeout=600,
    )


def load_out(qid):
    p = os.path.join(OUTPUT_DIR, f"{qid}_output.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def check_shape(qid):
    try:
        d = load_out(qid)
    except Exception as e:
        check(f"{qid} output exists+parses", False, str(e))
        return None
    ok = (set(d.keys()) == {"answer", "citations", "confidence"}
          and isinstance(d["answer"], str)
          and isinstance(d["citations"], list)
          and all(isinstance(c, str) for c in d["citations"])
          and isinstance(d["confidence"], (int, float)))
    check(f"{qid} JSON shape", ok, json.dumps(d)[:200])
    return d if ok else None


BACKEND = pick_backend()
print(f"backend={BACKEND}")

# fresh dirs
shutil.rmtree(TESTDIR, ignore_errors=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 0. chunking unit check (overlap correctness)
sys.path.insert(0, PROJ)
from chunking import chunk_text
words = [f"w{i}" for i in range(1000)]
ch = chunk_text(" ".join(words))
check("chunking produces >1 chunk", len(ch) > 1, f"got {len(ch)}")
if len(ch) > 1:
    a, b = ch[0].split(), ch[1].split()
    check("chunking overlap ~48 words", a[-48:] == b[:48])

# 1. index
r = run_app(["--index", CORPUS])
check("--index exit 0", r.returncode == 0, r.stderr[-500:])
check("index files written",
      all(os.path.exists(os.path.join(INDEX_DIR, f))
          for f in ("index.faiss", "chunks.jsonl", "meta.json")))

# 2. known-answer queries
r = run_app(["--corpus", CORPUS, "--query-id", "query_01",
             "--query", "What is the backup server hostname?"])
check("query_01 exit 0", r.returncode == 0, r.stderr[-300:])
d = check_shape("query_01")
if d:
    check("query_01 cites notes.txt", d["citations"] == ["notes.txt"], str(d["citations"]))
    check("query_01 answer contains hostname", "bkup-07.internal" in d["answer"],
          d["answer"][:200])

r = run_app(["--corpus", CORPUS, "--query-id", "query_02",
             "--query", "How many units of Widget X are in stock?"])
check("query_02 exit 0", r.returncode == 0, r.stderr[-300:])
d = check_shape("query_02")
if d:
    check("query_02 cites report.xlsx", d["citations"] == ["report.xlsx"], str(d["citations"]))
    check("query_02 answer contains 1337", "1337" in d["answer"], d["answer"][:200])

# 3. abstention path (forced via threshold=1.0 for determinism)
r = run_app(["--corpus", CORPUS, "--query-id", "query_03",
             "--query", "What is the capital of Atlantis?"],
            extra_env={"ABSTAIN_THRESHOLD": "1.0"})
check("query_03 exit 0", r.returncode == 0, r.stderr[-300:])
d = check_shape("query_03")
if d:
    check("query_03 abstains (empty answer, no citations)",
          d["answer"] == "" and d["citations"] == [],
          json.dumps(d)[:200])

# 4. withdrawn-document trap (heuristic gate)
print()
print("--- withdrawn-document tests ---")
from ingest.withdrawn import detect_withdrawn

# 4a. detector unit tests (deterministic, no backend needed)
detect_cases = [
    # (text, filename, expected)
    ("WITHDRAWN — superseded by Rev C. Do not use this document.", "", True),
    ("See Rev B (superseded) for the old procedure.", "", True),
    ("dataset v2 - WITHDRAWN", "", True),
    ("This document is obsolete.", "", True),
    ("Access revoked on 2026-01-01.", "", True),
    ("", "sop_RevA_withdrawn.pdf", True),
    ("Quarterly figures are final. Approved by finance.", "report.txt", False),
    # Calibrated 2026-10-08 (sample set): these former positives are too
    # noisy and are NOT withdrawal signals —
    ("Please use Rev C; this revision is deprecated.", "", False),
    ("Do not use tap water in the lab.", "safety.txt", False),
]
for text, fn, exp in detect_cases:
    got, snippet = detect_withdrawn(text, fn)
    check(f"detect_withdrawn text={text[:28]!r} file={fn!r}",
          got == exp, f"expected {exp}, got {got}")
    if exp:
        check(f"  marker snippet non-empty for {fn or text[:20]!r}",
              bool(snippet), "")

# 4b. end-to-end: withdrawn revision must never be cited
WD_CORPUS = os.path.join(TESTDIR, "wd_corpus")
WD_INDEX = os.path.join(TESTDIR, "wd_index")
shutil.rmtree(WD_CORPUS, ignore_errors=True)
shutil.rmtree(WD_INDEX, ignore_errors=True)
os.makedirs(WD_CORPUS, exist_ok=True)
os.makedirs(WD_INDEX, exist_ok=True)
with open(os.path.join(WD_CORPUS, "sop_RevB.txt"), "w", encoding="utf-8") as f:
    f.write("WITHDRAWN — superseded by Rev C. Do not use this document.\n"
            "Standard Operating Procedure Rev B.\n"
            "The emergency shutdown coolant pressure limit is 9000 kPa.\n")
with open(os.path.join(WD_CORPUS, "sop_RevC.txt"), "w", encoding="utf-8") as f:
    f.write("Standard Operating Procedure Rev C — current revision. "
            "Approved 2026-03-01.\n"
            "The emergency shutdown coolant pressure limit is 7500 kPa.\n")


def run_app_wd(args, extra_env=None):
    env = dict(os.environ)
    env["APP_INDEX_DIR"] = WD_INDEX
    env["APP_OUTPUT_DIR"] = OUTPUT_DIR
    env["EMB_BACKEND"] = BACKEND
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, os.path.join(PROJ, "app.py")] + args,
        cwd=PROJ, env=env, capture_output=True, text=True, timeout=600,
    )


r = run_app_wd(["--index", WD_CORPUS])
check("wd --index exit 0", r.returncode == 0, r.stderr[-500:])

with open(os.path.join(WD_INDEX, "meta.json"), encoding="utf-8") as f:
    wmeta = json.load(f)
wfiles = wmeta.get("files", {})
check("wd meta flags RevB withdrawn",
      wfiles.get("sop_RevB.txt", {}).get("withdrawn") is True, str(wfiles))
check("wd meta RevB marker snippet present",
      bool(wfiles.get("sop_RevB.txt", {}).get("marker")), str(wfiles))
check("wd meta RevC not withdrawn",
      wfiles.get("sop_RevC.txt", {}).get("withdrawn") is False, str(wfiles))

WD_QUERY = "What is the emergency shutdown coolant pressure limit?"
r = run_app_wd(["--corpus", WD_CORPUS, "--query-id", "wd_q1", "--query", WD_QUERY])
check("wd_q1 exit 0", r.returncode == 0, r.stderr[-300:])
d = check_shape("wd_q1")
if d:
    check("wd_q1 never cites withdrawn RevB",
          "sop_RevB.txt" not in d["citations"], str(d["citations"]))
    ok_answer = ((d["answer"] == "" and d["citations"] == [])
                 or ("7500" in d["answer"] and d["citations"] == ["sop_RevC.txt"]))
    check("wd_q1 abstains or answers from current RevC",
          ok_answer, json.dumps(d)[:300])
    check("wd_q1 exclusion logged to stderr",
          "excluded withdrawn document: sop_RevB.txt" in r.stderr,
          r.stderr[-300:])

# 4c. gate-off smoke test: WITHDRAWN_EXCLUDE=0 must not break querying
r = run_app_wd(["--corpus", WD_CORPUS, "--query-id", "wd_q2", "--query", WD_QUERY],
               extra_env={"WITHDRAWN_EXCLUDE": "0"})
check("wd_q2 exit 0 (gate off)", r.returncode == 0, r.stderr[-300:])
d = check_shape("wd_q2")
if d:
    check("wd_q2 cites one of the two docs",
          d["citations"] in (["sop_RevB.txt"], ["sop_RevC.txt"]),
          str(d["citations"]))
    check("wd_q2 no exclusion logged (gate off)",
          "excluded withdrawn document" not in r.stderr, r.stderr[-300:])

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL TESTS PASSED")
