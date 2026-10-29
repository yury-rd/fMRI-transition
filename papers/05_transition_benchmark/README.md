# HMM/VAR transition benchmark

Code and data for *Evaluating State-Transition Models in Short Resting-State
fMRI with Continuous-Dynamics Controls*.

Aleksandr Volkov, Yury Maximov, Tamoghna Chattopadhyay, Paul M. Thompson, and
Ildar Gabitov.

Under review at the [2027 IEEE International Symposium on Biomedical Imaging
(ISBI 2027)](https://biomedicalimaging.org/2027/).

The dataset has 216 latent realizations and four observation branches per
realization, giving 864 conditions. The generators are cyclic and reversible
three-state processes at two amplitudes, a common-factor AR process, and a
rotating AR process. Each condition has eight HMM and eight VAR candidates,
for 13,824 saved fits.

## Running the examples

Run these commands from the repository root after following the
[installation instructions](../../README.md#run-the-benchmark).

```sh
# Two-candidate executable example, compared with saved scores:
python papers/05_transition_benchmark/run.py --output results/transition_demo.json
# The complete 64-candidate menu for one latent realization and its four branches:
python papers/05_transition_benchmark/run.py --mode case --case cyclic_N97_A1.0_S101 --output results/transition_case.json
# All saved condition-level readouts, without refitting:
python papers/05_transition_benchmark/run.py --mode summary --output results/transition_results.json
```

The runner checks whether the generated observations exactly match the saved
arrays and reports the largest score difference from the saved candidate
table. Small floating-point differences can occur across platforms.
`--case` selects any production case ID in `data/config.json`; `--branch`
selects the observation branch for the two-candidate example. Running
`--mode case` for every production case reconstructs the full fit grid and
takes substantially longer than the example.

## Arrays and tables

`synthetic_216_cases.npz` is pickle-free. Keys are `<case_id>__latent`,
`<case_id>__hidden_state` for discrete generators, and `<case_id>__<branch>`
for `identity`, `stationary_filter`, `filter_change_at_test`, and `unit_change`.
Observation and latent arrays have shape N × 7. Hidden states have length N
and labels 0-2. `config.json` records the lengths, seeds, means, transitions,
observation filters, and candidate menus.

`candidate_scores.csv` contains validation and test log densities in nats
per frame, selection flags, and HMM flux for all fits. `readout.json` contains
the 864 condition rows and paired cyclic/reversible flux comparisons. The
four branches share a latent realization and are not independent draws.

## Scoring

Training ends at `int(.6*N)` and validation at `int(.8*N)`. Models and
observation-transform parameters fit training data only. Validation chooses
candidates. Test scores condition on all available past observations, with
the HMM filtering state carried across split boundaries. VAR uses the same
history convention.

Scores include the inverse-filter and training-standardization Jacobians
in the original observation units. HMM fitting uses 20 EM iterations. The
VAR evaluator preserves the study's 50-decimal-digit calculation. If a
candidate fails, the runner raises an exception.

The known generator labels support model comparisons. HMM flux is a
statistic of the fitted model; the continuous generators test whether it
can arise without discrete underlying states. No human time series are
included.
