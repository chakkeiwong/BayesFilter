# Observation-aware TT proposals: bounded mathematical and literature analysis

Question: how should the current observation influence Zhao–Cui TT fitting and
conditional proposals, and what role should a UKF guide play? The owner asked
to continue analysis on September 12 after a session incident investigation.
This stage develops a mathematical design and source comparison; it does not
launch a numerical campaign or change runtime code/defaults.

Research checkout: `/home/chakwong/BayesFilterZhaoCui`, branch
`zhao-cui-tt-regression-20260908`, with existing dirty work preserved. Notes
checkout: `/home/chakwong/BayesFilter`, branch `surrogate-hmc`.
Results: `docs/plans/artifacts/zhao-cui-observation-aware-tt-20260912-01/`.

## Evidence contract and skeptical audit

The mathematical target is the adjacent-state density in Zhao–Cui Algorithm 2,
including the observation likelihood. Algorithm 3 retains its TT conditional
proposal and exact importance correction. Gaussian guidance changes coordinates
or regression design, not the target or the importance denominator.

Compare the actual author Gaussian bridge and tempered nonlinear bridge, the
local unguided preparation, a standard UKF moment guide where applicable, and
likelihood-weighted cubature. Pass means an explicit change-of-variables and
regression-measure derivation, checked paper and author-code anchors, and clear
implementation gaps. No empirical superiority or production/HMC readiness can
follow from this stage. Formula counterexamples and inconsistent sampling or
weighting laws veto the corresponding correctness claims. Regression losses,
rank, ESS, and timing are not promotion evidence in this stage.

Skeptical audit: the existing likelihood factor is present, so “add the missing
observation likelihood” is a wrong problem statement. The sigma-point helper
is likelihood-reweighted cubature, not the standard UKF update. UKF is an owner
hypothesis, not an established best method. Preserve marginalization and
conditioning under the chosen chart; one-step Jacobian correctness alone does
not establish a recursively usable TT construction. Preserve support and the
actual numerical CDF law. These checks make the stage answer the active question.

No numerical defaults are promoted. Any covariance regularization, damping,
mixture, tempering schedule, TT rank, or row design remains a declared candidate.
For exact illustrative counterexamples, use a scalar Gaussian model and derive
its posterior algebraically; no stochastic performance claim is made.

Budget: one bounded source-and-derivation stage, up to 45 minutes of local
reading/research, at most 8 primary-source downloads and no GPU/training/HMC
jobs. Network publication/correction checks and citation discovery are authorized
by the literature request. Record unavailable metadata and unfinished coverage.
Stop this stage at a reviewable mathematical note and source/claim ledgers; do
not expand into implementation or experiments without a concrete next plan.

## Current checkpoint

Checked: the exact likelihood is present in local Algorithm 2 preparation;
the sigma-point helper has no non-test consumer. Zhao–Cui Section 5 already
describes Gaussian particle-moment bridging and tempered nonlinear bridging.
Its Algorithm 2 target and Algorithm 3 identity-ancestry update have been read.
Next: verify the author bridge call chain, derive the reference-density/row-law
correction, inspect the direct transport and regression literature, and save
the result with limitations. Detailed incident history stays outside this stage.
