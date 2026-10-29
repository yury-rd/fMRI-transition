import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from fmri_observations import observation, scoring, transitions

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "transition_benchmark/data"
RUNNER = ROOT / "transition_benchmark/run.py"
BRANCHES = {"identity", "stationary_filter", "filter_change_at_test", "unit_change"}


class TransitionBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)
        cls.config = json.loads((DATA / "config.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_all_generators_match_saved_arrays(self):
        cases = self.config["production_cases"]
        self.assertEqual(len(cases), 216)
        with np.load(DATA / "synthetic_216_cases.npz", allow_pickle=False) as saved:
            for case in cases:
                with self.subTest(case=case["id"]):
                    latent, hidden = transitions.simulate(case, self.config)
                    np.testing.assert_array_equal(latent, saved[case["id"] + "__latent"])
                    if hidden is not None:
                        np.testing.assert_array_equal(hidden, saved[case["id"] + "__hidden_state"])
                    branches = transitions.observe(latent, self.config, int(.8 * len(latent)))
                    self.assertEqual(set(branches), BRANCHES)
                    for branch, values in branches.items():
                        np.testing.assert_array_equal(values, saved[case["id"] + "__" + branch])

    def test_candidate_table_covers_every_condition(self):
        with (DATA / "candidate_scores.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 13824)
        actual = {(row["id"], row["branch"], row["candidate"]) for row in rows}
        candidates = {
            f"{family}{size}:{filter_name}"
            for family, sizes in [("HMM", self.config["states"]), ("VAR", self.config["var_orders"])]
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

    def test_conditional_scores_are_causal_and_unit_covariant(self):
        y = np.random.default_rng(101).normal(size=(97, 7))
        alpha = np.full(7, .35)
        v, jac = scoring.transform(y, alpha, 58)
        model = {"pi": [.5, .5], "trans": [[.8, .2], [.2, .8]],
                 "mean": np.zeros((2, 7)), "var": np.ones((2, 7))}
        scores = scoring.hmm_frames(v, model) + jac
        changed = y.copy()
        changed[-1] += 1000
        v2, jac2 = scoring.transform(changed, alpha, 58)
        np.testing.assert_array_equal(scores[:-1], (scoring.hmm_frames(v2, model) + jac2)[:-1])
        units = np.array([.125, .25, .5, 1, 2, 4, 8])
        v3, jac3 = scoring.transform(y * units, alpha, 58)
        np.testing.assert_allclose(scores - np.log(units).sum(), scoring.hmm_frames(v3, model) + jac3,
                                   rtol=0, atol=1e-10)

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
                cwd=directory, text=True, capture_output=True, check=True,
            )
            self.assertEqual(json.loads(completed.stdout)["benchmark"], "transition_benchmark")
            return json.loads(output.read_text())

    def test_demo_replays_scores_from_another_directory(self):
        result = self.run_cli("--branch", "stationary_filter")
        self.assertTrue(result["generator_exact_replay"])
        self.assertLessEqual(result["maximum_saved_score_error"], 1e-10)
        self.assertEqual(set(result["branches"]), {"stationary_filter"})
        self.assertEqual(set(result["branches"]["stationary_filter"]["candidates"]),
                         {"HMM3:none", "VAR1:none"})

    def test_summary_reads_saved_results_from_another_directory(self):
        result = self.run_cli("--mode", "summary")
        self.assertEqual(result, {"readout.json": json.loads((DATA / "readout.json").read_text())})

    def test_invalid_case_reports_configuration_path(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(RUNNER), "--case", "missing_case"],
                cwd=directory, text=True, capture_output=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(str(DATA / "config.json"), completed.stderr)


if __name__ == "__main__":
    unittest.main()
