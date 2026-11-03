import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from fmri_observations import observation, scoring, transitions

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "transition_benchmark/data"
CONFIG = ROOT / "transition_benchmark/config.json"
PRIVATE_TESTS = os.environ.get("FMRI_RUN_PRIVATE_TESTS") == "1"
RUNNER = ROOT / "transition_benchmark/run.py"
BRANCHES = {"identity", "stationary_filter", "filter_change_at_test", "unit_change"}


class TransitionBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)
        cls.config = json.loads(CONFIG.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    @unittest.skipUnless(
        PRIVATE_TESTS, "Set FMRI_RUN_PRIVATE_TESTS=1 with private data access"
    )
    def test_all_generators_match_saved_arrays(self):
        cases = self.config["production_cases"]
        self.assertEqual(len(cases), 216)
        with np.load(DATA / "synthetic_216_cases.npz", allow_pickle=False) as saved:
            for case in cases:
                with self.subTest(case=case["id"]):
                    latent, hidden = transitions.simulate(case, self.config)
                    np.testing.assert_array_equal(
                        latent, saved[case["id"] + "__latent"]
                    )
                    if hidden is not None:
                        np.testing.assert_array_equal(
                            hidden, saved[case["id"] + "__hidden_state"]
                        )
                    branches = transitions.observe(
                        latent, self.config, int(0.8 * len(latent))
                    )
                    self.assertEqual(set(branches), BRANCHES)
                    for branch, values in branches.items():
                        np.testing.assert_array_equal(
                            values, saved[case["id"] + "__" + branch]
                        )

    @unittest.skipUnless(
        PRIVATE_TESTS, "Set FMRI_RUN_PRIVATE_TESTS=1 with private data access"
    )
    def test_candidate_table_covers_every_condition(self):
        with (DATA / "candidate_scores.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 13824)
        actual = {(row["id"], row["branch"], row["candidate"]) for row in rows}
        candidates = {
            f"{family}{size}:{filter_name}"
            for family, sizes in [
                ("HMM", self.config["states"]),
                ("VAR", self.config["var_orders"]),
            ]
            for size in sizes
            for filter_name in self.config["filter_order"]
        }
        expected = {
            (case["id"], branch, candidate)
            for case in self.config["production_cases"]
            for branch in BRANCHES
            for candidate in candidates
        }
        self.assertEqual(len(actual), len(rows))
        self.assertEqual(actual, expected)

    def test_generated_controls_are_deterministic_and_preserve_unit_transform(self):
        for case in self.config["production_cases"]:
            with self.subTest(case=case["id"]):
                latent, hidden = transitions.simulate(case, self.config)
                repeated, repeated_hidden = transitions.simulate(case, self.config)
                np.testing.assert_array_equal(latent, repeated)
                self.assertEqual(latent.shape, (case["N"], 7))
                self.assertTrue(np.isfinite(latent).all())
                if hidden is not None:
                    np.testing.assert_array_equal(hidden, repeated_hidden)
                    self.assertTrue(set(hidden).issubset({0, 1, 2}))
                branches = transitions.observe(
                    latent, self.config, int(0.8 * len(latent))
                )
                self.assertEqual(set(branches), BRANCHES)
                np.testing.assert_array_equal(branches["identity"], latent)
                np.testing.assert_allclose(
                    branches["unit_change"],
                    branches["stationary_filter"] * self.config["unit_scale"],
                    rtol=0,
                    atol=0,
                )

    def test_conditional_scores_are_causal_and_unit_covariant(self):
        y = np.random.default_rng(101).normal(size=(97, 7))
        alpha = np.full(7, 0.35)
        v, jac = scoring.transform(y, alpha, 58)
        model = {
            "pi": [0.5, 0.5],
            "trans": [[0.8, 0.2], [0.2, 0.8]],
            "mean": np.zeros((2, 7)),
            "var": np.ones((2, 7)),
        }
        scores = scoring.hmm_frames(v, model) + jac
        changed = y.copy()
        changed[-1] += 1000
        v2, jac2 = scoring.transform(changed, alpha, 58)
        np.testing.assert_array_equal(
            scores[:-1], (scoring.hmm_frames(v2, model) + jac2)[:-1]
        )
        units = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])
        v3, jac3 = scoring.transform(y * units, alpha, 58)
        np.testing.assert_allclose(
            scores - np.log(units).sum(),
            scoring.hmm_frames(v3, model) + jac3,
            rtol=0,
            atol=1e-10,
        )

    def test_observation_parameters_use_training_data_only(self):
        y = np.random.default_rng(211).normal(size=(97, 7))
        original = observation.alpha_candidates(y, 58)
        changed = y.copy()
        changed[58:] *= 1000
        updated = observation.alpha_candidates(changed, 58)
        for name in original:
            np.testing.assert_array_equal(original[name], updated[name])

    def run_cli(self, *args):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested/result.json"
            completed = subprocess.run(
                [sys.executable, str(RUNNER), *args, "--output", str(output)],
                cwd=directory,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(
                json.loads(completed.stdout)["benchmark"], "transition_benchmark"
            )
            return json.loads(output.read_text())

    def test_demo_generates_scores_from_another_directory(self):
        result = self.run_cli("--branch", "stationary_filter")
        self.assertFalse(result["saved_verification_requested"])
        self.assertIsNone(result["generator_exact_replay"])
        self.assertIsNone(result["maximum_saved_score_error"])
        for fitted in result["branches"]["stationary_filter"]["candidates"].values():
            self.assertTrue(np.isfinite([fitted["validation"], fitted["test"]]).all())
        self.assertEqual(set(result["branches"]), {"stationary_filter"})
        self.assertEqual(
            set(result["branches"]["stationary_filter"]["candidates"]),
            {"HMM3:none", "VAR1:none"},
        )

    @unittest.skipUnless(
        PRIVATE_TESTS, "Set FMRI_RUN_PRIVATE_TESTS=1 with private data access"
    )
    def test_summary_reads_saved_results_from_another_directory(self):
        result = self.run_cli("--mode", "summary")
        self.assertEqual(
            result, {"readout.json": json.loads((DATA / "readout.json").read_text())}
        )

    def test_private_modes_report_missing_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            for args in [["--mode", "summary"], ["--verify-saved"]]:
                completed = subprocess.run(
                    [sys.executable, str(RUNNER), *args, "--data-dir", directory],
                    cwd=directory,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    "git submodule update --init transition_benchmark/data",
                    completed.stderr,
                )

    def test_summary_loader_with_runtime_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {"scope": "synthetic loader test", "rows": []}
            (Path(directory) / "readout.json").write_text(json.dumps(payload))
            result = self.run_cli("--mode", "summary", "--data-dir", directory)
            self.assertEqual(result, {"readout.json": payload})

    def test_invalid_case_reports_configuration_path(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(RUNNER), "--case", "missing_case"],
                cwd=directory,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(str(CONFIG), completed.stderr)


if __name__ == "__main__":
    unittest.main()
