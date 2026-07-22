# PP-UKF Reasonable-Epsilon Runtime Diagnostic

Date: 2026-07-21

## Question And Evidence Contract

Determine why the repaired PP-UKF tuning-only retry remained in
`windowed_mass_operational_warmup_start` before producing a result. The exact
comparator is the same frozen PP-UKF transport and target used by the retry.
The primary diagnostic is separate wall time for one target evaluation, HMC
bootstrap, and one proposal. Finite values and a finite proposal are required;
they are engineering diagnostics, not tuning or scientific promotion criteria.

The artifact is
`docs/plans/artifacts/bayesfilter-pp-ukf-epsilon-probe-20260721-01/result.json`.
No sampling, claim run, or posterior interpretation is authorized by this
diagnostic.

## Skeptical Audit And Premortem

- Wrong baseline risk: using a different transport or target would not explain
  the retry. The command binds the existing frozen transport SHA-256 and the
  PP-UKF target signature.
- Proxy risk: finite values or a single fast proposal do not establish HMC
  validity. They are explanatory diagnostics only.
- Environment risk: an untrusted CUDA failure is not interpreted as a driver
  failure; the probe is rerun with trusted GPU access and memory growth.
- Harness risk: rank-2 PP-UKF target inputs are bound through the repository
  batch-native adapter, matching the real tuner.
- Repair risk: changing eager defaults or bypassing epsilon qualification would
  alter the scientific contract. The repair only propagates the existing XLA
  choice into epsilon proposals and preserves the default eager path.

## Decision Rule

If the proposal is finite but materially expensive, use the localized compiled
epsilon-search repair, run focused regression tests, and only then consider a
fresh tuning-only retry under the existing target, route, budget, and holdout
contract. If the probe is non-finite or fails before proposal completion, stop
and classify the failure before any tuning retry.

## Outcome

The proposal and compiled epsilon-search probes were finite. The repair was
implemented and the focused suite passed, but the subsequent fresh tuning-only
retry terminated inside the first operational window without a terminal
artifact. This plan therefore closes with the runtime diagnosis validated and
the full tuning result still unresolved.
