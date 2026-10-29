# fMRI-transition

Code and data for *Evaluating State-Transition Models in Short Resting-State fMRI
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
model (VAR). It checks the generated observations against the saved arrays
and compares the scores with the recorded results. The JSON output includes
`generator_exact_replay` and `maximum_saved_score_error`. Small floating-point
differences can occur across platforms.

## Run the benchmark

Fit all 64 candidates for one latent realization across its four observation
branches:

```sh
python transition_benchmark/run.py --mode case --case cyclic_N97_A1.0_S101 --output results/transition_case.json
```

Export all saved condition-level results without refitting:

```sh
python transition_benchmark/run.py --mode summary --output results/transition_results.json
```

Use `--case` to choose a production case from
[config.json](transition_benchmark/data/config.json). In demo mode, `--branch`
selects `identity`, `stationary_filter`, `filter_change_at_test`, or `unit_change`.
The [benchmark notes](transition_benchmark/README.md) describe the data format
and scoring conventions.

## Repository layout

| Directory | Contents |
| --- | --- |
| [transition_benchmark/](transition_benchmark/) | Runner, synthetic data, configuration, and saved results |
| [fmri_observations/](fmri_observations/) | Generators, observation transforms, HMM and VAR fitting, and conditional scoring |
| [tests/](tests/) | Generator replay, scoring, and command-line checks |

The installation above makes `fmri_observations` available to Python. Output
files go to `results/`, which is excluded from version control.

## Experiments

The dataset contains 216 latent realizations, each with four observation
branches: identity, stationary filtering, a filter change at the test boundary,
and a change of units. The resulting 864 conditions have eight HMM and eight
VAR candidates each, for 13,824 saved fits. All time series are synthetic.

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

The tests check every saved generator realization, the candidate table, causal
scoring, and execution from outside the repository directory. To run those
checks together with all three examples and regenerate the verification report:

```sh
python tests/verify_reproduction.py
```

The report is saved to
[transition_benchmark/VERIFICATION.json](transition_benchmark/VERIFICATION.json).

## License and citation

Code and synthetic data are available under the [MIT license](LICENSE).
See [CITATION.md](CITATION.md) for the paper citation and
[DATA_LICENSES.md](DATA_LICENSES.md) for data details.
