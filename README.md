# fMRI-transition

Code for *Evaluating State-Transition Models in Short Resting-State fMRI
with Continuous-Dynamics Controls*.

Aleksandr Volkov, Yury Maximov, Tamoghna Chattopadhyay, Paul M. Thompson, and
Ildar Gabitov.

Under review at the [2027 IEEE International Symposium on Biomedical Imaging
(ISBI 2027)](https://biomedicalimaging.org/2027/).

## Quick start

Requires Python 3.11 or later. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install --no-deps -e .
python transition_benchmark/run.py --output results/example.json
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The example fits one hidden Markov model (HMM) and one vector autoregressive
model (VAR) on arrays generated in memory. The public demo and tests require
no data download. To compare against saved arrays and scores, authorized users
can initialize the private companion and pass `--verify-saved` as described below.
Without that flag, replay fields are `null`, so generated output does not claim
to have verified the private records.

## Run the benchmark

Fit all 64 candidates for one latent realization across its four observation
branches:

```sh
python transition_benchmark/run.py --mode case --case cyclic_N97_A1.0_S101 --output results/transition_case.json
```

## Private saved data

Saved arrays, score tables, readouts, and historical verification reports live
in [yury-rd/fMRI-transition-data](https://github.com/yury-rd/fMRI-transition-data).
They are absent from public source history and Python distributions. Authorized
users can check out the companion at `transition_benchmark/data/`:

```sh
git submodule update --init transition_benchmark/data
python transition_benchmark/run.py --verify-saved --output results/replay.json
python transition_benchmark/run.py --mode summary --output results/transition_results.json
```

These commands require GitHub authentication and access to the private companion.
Missing files produce an error explaining how to initialize it. `--data-dir`
can point to another authorized checkout. `--verify-saved` compares generated
observations and fitted scores with the saved arrays and candidate table;
`--mode summary` reads saved condition-level results without refitting. Small
floating-point differences can occur across platforms.

Use `--case` to choose a production case from
[config.json](transition_benchmark/config.json). In demo mode, `--branch`
selects `identity`, `stationary_filter`, `filter_change_at_test`, or `unit_change`.
The [benchmark notes](transition_benchmark/README.md) describe the data format
and scoring conventions.

## Repository layout

| Directory | Contents |
| --- | --- |
| [transition_benchmark/](transition_benchmark/) | Runner and public generator configuration; optional private `data/` submodule |
| [fmri_observations/](fmri_observations/) | Generators, observation transforms, HMM and VAR fitting, and conditional scoring |
| [tests/](tests/) | Generated controls, scoring, command-line checks, and opt-in private replay |

The installation above makes `fmri_observations` available to Python. Output
files go to `results/`, which is excluded from version control.

## Experiments

The generator configuration defines 216 latent realizations, each with four observation
branches: identity, stationary filtering, a filter change at the test boundary,
and a change of units. The resulting 864 conditions have eight HMM and eight
VAR candidates each, for 13,824 candidate fits. All generated time series are synthetic.

The generators include cyclic and reversible three-state processes at two
amplitudes, a common-factor AR process, and a rotating AR process. Models fit
the first 60% of each series; validation uses the next 20%, and testing uses
the final 20%. Candidate selection uses validation scores. Test predictions
condition on the full observed past.

Fitted HMM flux alone does not establish that the underlying physiology has
discrete states. The continuous generators provide controls for that
interpretation.

## Checks

```sh
python -m unittest discover -s tests -v
```

The default tests check deterministic generated controls, causal scoring,
missing-private-data errors, and execution from outside the repository. Private
replay tests are skipped by default. After initializing the private companion,
run the full saved-data comparison and generate a local verification report:

```sh
python tests/verify_reproduction.py
```

The report is saved to `results/verification.json`, outside version control.
To enable only the private tests, use `FMRI_RUN_PRIVATE_TESTS=1 python -m unittest
discover -s tests -v` (set the environment variable separately on Windows).

## License and citation

Public code is available under the [MIT license](LICENSE).
See [CITATION.md](CITATION.md) for the paper citation and
[DATA_LICENSES.md](DATA_LICENSES.md) for data details.
