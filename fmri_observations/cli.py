"""Run generated controls, optionally comparing with private saved results."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def private_file(data, name):
    """Require an authorized private checkout for saved-data operations."""
    path = data / name
    if not path.is_file():
        raise ValueError(
            f"Private data file missing: {path}. With access to "
            "yury-rd/fMRI-transition-data, run from the repository root: "
            "git submodule update --init transition_benchmark/data. "
            "Generated demo/case modes need no private data."
        )
    return path


def transition_demo(folder, args):
    from . import observation
    from . import transitions as model

    config_path = folder / "config.json"
    config = json.loads(config_path.read_text())
    cases = [case for case in config["production_cases"] if case["id"] == args.case]
    if not cases:
        raise ValueError(f"Unknown case; see {config_path}")
    case = cases[0]
    latent, _ = model.simulate(case, config)
    cut, end = int(0.6 * len(latent)), int(0.8 * len(latent))
    branches = model.observe(latent, config, end)
    exact = None
    baselines = {}
    if args.verify_saved:
        arrays = private_file(args.data_dir, "synthetic_216_cases.npz")
        table = private_file(args.data_dir, "candidate_scores.csv")
        with np.load(arrays, allow_pickle=False) as saved:
            exact = all(
                np.array_equal(values, saved[case["id"] + "__" + name])
                for name, values in branches.items()
            )
        with table.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["id"] == case["id"]:
                    baselines[(row["branch"], row["candidate"])] = row

    selected_branches = list(branches) if args.mode == "case" else [args.branch]
    output = {}
    max_error = 0.0 if args.verify_saved else None
    for name in selected_branches:
        values = branches[name]
        alphas = observation.alpha_candidates(values, cut)
        fits = {}
        states = config["states"] if args.mode == "case" else [3]
        orders = config["var_orders"] if args.mode == "case" else [1]
        filters = config["filter_order"] if args.mode == "case" else ["none"]
        for states_count in states:
            for filter_name in filters:
                fits[f"HMM{states_count}:{filter_name}"] = model.hmm_candidate(
                    values, alphas[filter_name], states_count, case["seed"], cut, end
                )
        for order in orders:
            for filter_name in filters:
                fits[f"VAR{order}:{filter_name}"] = model.var_candidate(
                    values, alphas[filter_name], order, cut, end
                )
        selected_hmm = model.select(
            fits, [key for key in fits if key.startswith("HMM")]
        )
        selected_var = model.select(
            fits, [key for key in fits if key.startswith("VAR")]
        )
        candidates = {}
        for key, fitted in fits.items():
            candidates[key] = {split: fitted[split] for split in ["validation", "test"]}
            if "flux" in fitted:
                candidates[key]["flux"] = fitted["flux"]
            if args.verify_saved:
                max_error = max(
                    max_error,
                    *(
                        abs(fitted[split] - float(baselines[(name, key)][split]))
                        for split in ["validation", "test"]
                    ),
                )
        output[name] = {
            "candidates": candidates,
            "selected_hmm": selected_hmm,
            "selected_var": selected_var,
            "hmm_minus_var_test": fits[selected_hmm]["test"]
            - fits[selected_var]["test"],
        }
    return {
        "case": case,
        "saved_verification_requested": args.verify_saved,
        "generator_exact_replay": exact,
        "maximum_saved_score_error": max_error,
        "menu": "full" if args.mode == "case" else "two-candidate smoke example",
        "branches": output,
    }


def main(benchmark_folder, argv=None):
    folder = Path(benchmark_folder).resolve()
    parser = argparse.ArgumentParser(
        description="Compare HMM and VAR models on generated synthetic fMRI observations."
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "case", "summary"],
        default="demo",
        help="demo: two fits; case: all 64 fits; summary: private saved readouts",
    )
    parser.add_argument("--output", type=Path, help="write JSON to this path")
    parser.add_argument(
        "--case",
        default="cyclic_N97_A1.0_S101",
        help="production case ID from transition_benchmark/config.json",
    )
    parser.add_argument(
        "--verify-saved",
        action="store_true",
        help="compare generated arrays and fitted scores with the private data companion",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=folder / "data",
        help="private data checkout (default: transition_benchmark/data)",
    )
    parser.add_argument(
        "--branch",
        choices=[
            "identity",
            "stationary_filter",
            "filter_change_at_test",
            "unit_change",
        ],
        default="identity",
        help="observation branch for demo mode",
    )
    args = parser.parse_args(argv)
    if args.mode == "summary" and args.verify_saved:
        parser.error("--verify-saved requires demo or case mode")
    try:
        if args.mode == "summary":
            result = {
                "readout.json": json.loads(
                    private_file(args.data_dir, "readout.json").read_text()
                )
            }
        else:
            with threadpool_limits(limits=1):
                result = transition_demo(folder, args)
        payload = json.dumps(plain(result), indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "benchmark": folder.name,
                        "mode": args.mode,
                    }
                )
            )
        else:
            print(payload, end="")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
