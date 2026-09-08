# Phase 25 q=20 LEDH Density-Adapter and Target-Measure Probe

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `DIRECT_FULL_STATE_LEDH_BLOCKED_SINGULAR_MEASURE`  
Budget cap: `3600 s` within the unchanged `64800 s` campaign cap  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase25`

## Research question

Can the actual q=20 SSL-LSTM target support an explicit LEDH-PFPF density
adapter in its declared measure, or does the structural transition become a
singular state-space measure that requires a different reduced-coordinate
proposal?

This phase tests the smallest repair indicated by Phase 24. It does not change
the target, replace the parameter particle authority, or call an affine map
LEDH. A reduced-coordinate construction is an extension until its target and
Jacobian identity are demonstrated.

## Evidence contract

| Item | Predeclared rule |
|---|---|
| Comparator | q=20 `BatchNativeSSLLSTMComplexityPosteriorTarget` with the exact Phase 24 source/target signature |
| Primary criterion | finite explicit Gaussian innovation log density, measured rank of induced state transition covariance, and an explicit dimension/measure compatibility decision |
| Promotion veto | nonfinite covariance/eigenvalue, source/hash mismatch, or a claim that a singular state transition has an ordinary full-state Lebesgue density |
| Continuation veto | target identity unavailable/contradictory, or no reduced coordinate can carry a positive proposal density on the required support |
| Explanatory diagnostics | aggregate UKF value/score, residual norms, eigenvalue spectrum, and dimensions |
| Nonclaims | no q=20 LEDH admission, posterior correctness, mode discovery, IID whitening, HMC readiness, or statistical ranking |
| Artifact | unique JSON/Markdown result with source hashes, command, environment, exact tensors, rank receipt, and refreshed Phase 26 decision |

## Required calculations

1. Instantiate the q=20 target in the CPU/reference lane and bind the Phase 24
   structural callbacks.
2. Evaluate the transition innovation Jacobian `G` and the induced covariance
   `G Q G^T` at fixed deterministic points. Record eigenvalues and numerical
   rank using a declared diagnostic tolerance; the tolerance is a diagnostic
   hypothesis, not a promotion threshold.
3. Evaluate a reduced innovation Gaussian log density for realized transition
   innovations using the declared `Q`; verify finiteness and positive
   definiteness without pretending it is a density over all 60 state
   coordinates.
4. Compare the 60-dimensional structural state measure and 20-dimensional
   innovation measure with the four-dimensional q=20 parameter target. Record
   whether a common full-dimensional LEDH proposal can be bound without a
   target change.
5. If the direct route is incompatible, refresh Phase 26 to investigate a
   reduced innovation-coordinate flow. If it is compatible (only if all terms
   are actually identified), refresh Phase 26 to a source-bound adapter fixture.

## Assumption/default audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| CPU-hidden TensorFlow | repository reference-lane policy | cannot support GPU claim | device receipt | reference exception |
| q=20, Phase 24 signature | target identity and preceding artifact | stale target | source/hash parity | required |
| rank tolerance `1e-10` | numerical diagnostic placeholder | rank label changes near zero | report full spectrum and rerun sensitivity | hypothesis only |
| fixed test points | deterministic audit convenience | misses state-dependent rank change | evaluate multiple points and record them | diagnostic only |
| reduced innovation density | Gaussian `Q` field in structural contract | does not identify parameter posterior | explicit measure ledger | candidate repair, not default |

## Pre-mortem

- A finite innovation density could be mistaken for a full-state density. The
  rank receipt and singular-support statement prevent that interpretation.
- A state-space flow could be implemented while the particle authority remains
  four-dimensional parameter space. The dimension ledger blocks silent measure
  substitution.
- Eigenvalue rank could depend on a convenient tolerance. The full spectrum and
  sensitivity table remain descriptive; no promotion rests on the tolerance.
- An aggregate UKF score could be relabeled as a decomposed transition plus
  observation likelihood. The adapter must report aggregate-only status unless
  exact equality is shown.

## Exact command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_ledh_density_adapter_probe_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase25-attempt1
```

## Inter-phase repair/refresh

After the command, run the focused unit tests and inspect the JSON schema.
Classify failures as harness, numerical, target/interface, or scientific
measure mismatch. Repair only within the same target, scope, and budget. A
singular full-state transition is a route-specific blocker for direct LEDH,
not a blocker for the wider ETPF/SMC/NeuTra investigation; refresh the next
subplan accordingly.

## Executed receipt

The prescribed command completed in `7.5 s`. The induced covariance had rank
`20` for state dimension `60` at all tested points and tolerances; the
innovation density and aggregate target were finite. Direct full-state LEDH
was blocked by the singular measure and refreshed to Phase 26.
