# fMRI-transition

A synthetic benchmark for comparing hidden Markov models (HMMs) and vector
autoregressive models (VARs) in short resting-state fMRI. The experiments test
how observation filters and changes in measurement units affect predictive
scores and estimated state-transition flux.

This repository contains the code and data for *Evaluating State-Transition
Models in Short Resting-State fMRI with Continuous-Dynamics Controls*, from
[fMRI Observation Benchmarks v1.0.0](https://github.com/cornada/fmri-observation-benchmarks/tree/v1.0.0/papers/05_transition_benchmark).
It includes the shared Python library needed to run the benchmark.

## Run the benchmark

Requires Python 3.11 or later. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install --no-deps -e .
python papers/05_transition_benchmark/run.py --output results/example.json
```

The example fits one HMM and one VAR, checks the generated observations against
the saved arrays, and compares the scores with the released results. Small
floating-point differences can occur across platforms.

To fit all 64 candidates across the four observation branches for one latent
realization:

```sh
python papers/05_transition_benchmark/run.py --mode case --case cyclic_N97_A1.0_S101 --output results/transition_case.json
```

To export the saved results without refitting, or run the shared library tests:

```sh
python papers/05_transition_benchmark/run.py --mode summary --output results/transition_results.json
python -m unittest discover -s tests -v
```

## Data and methods

The release contains 216 latent realizations, each with four observation
branches: identity, stationary filtering, a filter change at the test boundary,
and a change of units. The resulting 864 conditions have eight HMM and eight
VAR candidates each, for 13,824 saved fits. All time series are synthetic.

The generators include cyclic and reversible three-state processes at two
amplitudes, a common-factor AR process, and a rotating AR process. Models fit
the first 60% of each series; validation uses the next 20%, and testing uses
the final 20%. Candidate selection uses validation scores. Test predictions
condition on the full observed past.

The [benchmark notes](papers/05_transition_benchmark/README.md) describe the
array layout, scoring conventions, and available run options. Data and saved
scores are in [papers/05_transition_benchmark/data](papers/05_transition_benchmark/data);
the shared implementation is in [src/fmri_observation](src/fmri_observation).
Fitted HMM flux alone does not establish that the underlying physiology has
discrete states; the continuous generators provide controls for that interpretation.

## Source and license

Copied from the paper-specific v1.0.0 release of
[cornada/fmri-observation-benchmarks](https://github.com/cornada/fmri-observation-benchmarks).
The benchmark code and data retain the upstream implementation; this copy
updates the repository description and documentation.

Code and synthetic data are available under the [MIT license](LICENSE).
See [CITATION.md](CITATION.md) for the source citation and
[DATA_LICENSES.md](DATA_LICENSES.md) for data details.

`SOURCE_PROVENANCE.json` and `VERIFICATION.json` preserve the upstream
source records and verification report for the original five-paper release.
