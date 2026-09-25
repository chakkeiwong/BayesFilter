# BayesFilter

The official guide is built from [docs/main.tex](docs/main.tex), which includes
the chapters under `docs/chapters/`. Its compiled local copy is `docs/main.pdf`.
From the repository root, build it with:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Guide updates must refresh `docs/main.pdf`. Temporary build directories are
internal build details; any PDFs retained under experiment artifacts are dated
historical records. Reader-facing links should point to the official source or
`docs/main.pdf`.

## Where to start

- Pruned second-order direct-factor SRUKF: `docs/reference/pruned-direct-factor-srukf.md` and the square-root sigma-point guide chapter.
- HMC interfaces and candidate-set validation: `docs/chapters/ch21b_hmc_tuning_interfaces.tex`
- NeuTra and fixed-transport restart selection: `docs/chapters/ch26b_neutra_transport_hmc.tex`
- Runnable candidate-selection example: `docs/examples/fixed_transport_candidate_selection.py`

The ordinary and supported fixed-transport tuners use one shared candidate-set
controller. Each candidate receives its own acceptance and numerical-health
checks and fresh verification; every verified pair is retained. R-hat, ESS and
MCSE belong to separate posterior assessment and do not qualify or rank tuning
candidates. The older
`select_fixed_transport_candidate_set` export remains diagnostic-only for
historical payloads; it cannot schedule the active procedure or issue a tuning
handoff.
