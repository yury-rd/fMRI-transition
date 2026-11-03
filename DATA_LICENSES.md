# Data access and license

Saved synthetic observations, latent trajectories, hidden-state labels,
candidate scores, and aggregate readouts are stored in the private
[yury-rd/fMRI-transition-data](https://github.com/yury-rd/fMRI-transition-data)
companion, mounted at `transition_benchmark/data/` for authorized users.
They are not bundled in public source or Python distributions.

The public generator configuration and code can produce synthetic observations
at runtime. The retained license for the saved synthetic data is MIT; access
to the private companion is managed separately. This benchmark contains no
human time series or participant-level records.
