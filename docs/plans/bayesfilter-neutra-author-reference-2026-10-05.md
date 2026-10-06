# Bounded Gabrié original-code reference

This implements the original-author track of the authorized remedy plan. It
calls the preserved `flonaco.training.train` and `RealNVP_MLP` unchanged in an
isolated reference process. It is not a BayesFilter implementation, an IAF fit,
a default backend change, or a reproduction of the complete Figure 5 suite.

Appendix F and G.1 of the locally stored paper specify six coupling pairs,
conditioner depth three/width 100, two unit-covariance components separated by
10 with weights 1:2, 40 walkers, ten updates per walker, batch 400, 1,500 Adam
steps and LR .005. Anchors: `gabrie-adaptive-flows.layout.txt:1614` and `:1705`
under `.localresources/fab-coverage-followup-20260928/`. The original source
anchors under `.localresources/flonaco-author-20260929/upstream/` are
`flonaco/training.py:24`, `:96`, `:212`, `flonaco/real_nvp_mlp.py:78` and
`experiments/gaussian/run_training_mog.py:28`.

Set centers (-5,0)/(5,0) as the symmetric realization of the stated separation;
do not infer an exact coordinate reconstruction from the paper description.
Initialization scale 1e-6, local step argument 1e-4 (effective 2e-4 after the
source's dimension multiplier), 100 burn-in steps, jump tolerance 100 and clip
norm 10,000 come from the source experiment/defaults, not verified Figure 5
settings. They are explicit source-realization hypotheses. Preserve the author's
known-mode initialization, global-MH/local-ULA order, retries, clipping and
optimizer behavior. ULA has discretization error; do not relabel it exact MALA.

The question is whether this complete original controller executes, how long
it takes and what raw-flow/mode-coverage evidence it delivers under these stated
settings. Full training completion and finite parameters are mechanics checks.
Independent exact samples assess forward KL and raw-flow samples assess reverse
KL, responsibility weights and off-component probability. These are descriptive
single-run results, not evidence of superiority, posterior convergence, generic
mode discovery or exact equivalence to the published figure. An unchanged local
IAF on other targets is not a matched comparator.

The installed `mathdevmcp-backends` environment contains PyTorch CPU, NumPy,
Matplotlib and SciPy; the CUDA environment lacks plotting dependencies. Use the
existing CPU environment as an explicit original-code reference exception with
one Torch/BLAS thread and CUDA intentionally hidden before import. It does not
replace GPU/XLA canonical IAF training. No packages or environments are changed.

First run 100 source iterations to establish compatibility and measured pricing;
this is a cost/engineering diagnostic, not a shortened substitute for 1,500.
Seed 1705 is a prospective convenience choice. The pilot uses a 180-second
wall/CPU cap (engineering containment, no scientific interpretation of timeout).
If completed, conservatively project 15 times its total process cost and price
the full run against the remaining CPU ledger after reserving the active IAF
comparison. Full-run seed 1706 is separate, with fresh initialization. At most
one full attempt and one localized compatibility retry fit within the existing
allocation; each has a fresh output directory. Neither elapsed budget nor source
failure changes the scientific target or permits discarding failed evidence.

Run manifests include the interpreter, exact command, Git revision, upstream
file hashes, seed, source settings, intentionally hidden GPUs and actual CPU/wall
cost. Use the output root `artifacts/neutra-source-fit-remedy-2026-10-04/author-reference/`
and settle its costs into the scientific/shared ledger after the active worker
controller releases its lock. Until settlement they remain reserved, not free.

Skeptical audit: this is a separately labeled source realization with all known
paper settings, not claimed exact experimental reproduction. The original
controller can distinguish missing local procedure from primitive parity but
cannot isolate architecture or hardware relative to the local IAF campaign.
The pilot cannot support a quality claim. The bounded diagnostic is justified
without a package installation or another adaptive-controller port.

Pricing completed: the unchanged controller ran all 100 diagnostic iterations
with no compatibility patch in 18.860 process seconds / 18.442 CPU-core seconds,
including imports, ten original density plots and independent endpoint checks.
The conservative 15-fold complete-process projection is 282.894 seconds; the
full-run ceiling rounds this up to 283 seconds. This and the pilot fit comfortably
inside the CPU remainder after the complete IAF comparison reservation. The
launcher's 1,800-second source-reference ceiling is only an engineering reservation
limit; this measured run requests 283. Source settings stay unchanged for the
full 1,500-step seed-1706 attempt. No quality inference is made from the pilot.
