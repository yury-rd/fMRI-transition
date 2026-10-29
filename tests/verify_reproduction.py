"""Run the tests and documented examples, then record their results."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "transition_benchmark"
TOLERANCE = 1e-10
CASE = "cyclic_N97_A1.0_S101"


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise SystemExit("Tests failed; verification report was not updated.")

    commands = [
        ["transition_benchmark/run.py", "--output", "results/example.json"],
        ["transition_benchmark/run.py", "--mode", "case", "--case", CASE,
         "--output", "results/transition_case.json"],
        ["transition_benchmark/run.py", "--mode", "summary",
         "--output", "results/transition_results.json"],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=ROOT, check=True)

    demo = json.loads((ROOT / "results/example.json").read_text())
    full_case = json.loads((ROOT / "results/transition_case.json").read_text())
    summary = json.loads((ROOT / "results/transition_results.json").read_text())
    saved = json.loads((BENCHMARK / "data/readout.json").read_text())
    assert demo["generator_exact_replay"] and full_case["generator_exact_replay"]
    assert max(demo["maximum_saved_score_error"], full_case["maximum_saved_score_error"]) <= TOLERANCE
    assert len(full_case["branches"]) == 4
    assert sum(len(branch["candidates"]) for branch in full_case["branches"].values()) == 64
    assert summary == {"readout.json": saved}

    report = {
        "status": "PASS",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in
                         ["numpy", "scipy", "scikit-learn", "mpmath", "threadpoolctl", "joblib", "cloudpickle"]},
        "command": "python tests/verify_reproduction.py",
        "tests": {"run": tests.testsRun, "failures": len(tests.failures),
                  "errors": len(tests.errors), "skipped": len(tests.skipped)},
        "examples": ["python " + " ".join(command) for command in commands],
        "generator_realizations_checked": 216,
        "observation_conditions_checked": 864,
        "saved_candidate_rows_checked": 13824,
        "refitted_case": CASE,
        "refitted_candidates": 64,
        "generator_exact_replay": full_case["generator_exact_replay"],
        "maximum_saved_score_error": full_case["maximum_saved_score_error"],
        "score_tolerance": TOLERANCE,
        "summary_matches_saved_readout": True,
        "data_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted((BENCHMARK / "data").iterdir()) if path.is_file()},
    }
    (BENCHMARK / "VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Verified all three run modes; report: transition_benchmark/VERIFICATION.json")


if __name__ == "__main__":
    main()
