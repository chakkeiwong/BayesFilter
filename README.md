# BayesFilter

## Where to start

- HMC interfaces and candidate-set validation: `docs/chapters/ch21b_hmc_tuning_interfaces.tex`
- NeuTra and fixed-transport restart selection: `docs/chapters/ch26b_neutra_transport_hmc.tex`
- Runnable candidate-selection example: `docs/examples/fixed_transport_candidate_selection.py`

The candidate-set API validates frozen fixed-transport candidates in staged,
candidate-local rungs, retains every candidate that passes the declared hard
screens, and applies a caller-supplied nomination score. R-hat and ESS are
viability diagnostics, not universal ranking scores; a descriptive nomination
does not establish statistical superiority.
