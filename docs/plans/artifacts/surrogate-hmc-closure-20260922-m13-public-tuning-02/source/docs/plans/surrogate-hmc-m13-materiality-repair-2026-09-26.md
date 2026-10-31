# M13 bounded control repair — 2026-09-26

Goal: test surrogate HMC. This is the already authorized master's control-repair
phase, after materiality-controls02 selected no common calibration/validation
pair. It is neither posterior qualification nor a change of target or direction.

## Prospective evidence contract and audit

Keep the endpoint, full-support normal prior, M13 T20/N4096, FP32/TF32/GPU/XLA
canonical analytical force, data1704 exploratory pilot, and .01 maximum matched
acceptance-probability tolerance unchanged. Freeze a new weaker geometric family
(.001, .0001, .00001, .000001), all positive. Calibration data1706/design15 and
validation2706/design16 are fresh; previous validation is not reused for selection.
The strongest arm matches the preceding family's weakest arm in controls only.
The first adjacent pair passing both datasets nominates its stronger member.
No current claim uses data3701.

The same canonical implementation receives these settings. Relative to the
original controls, reset ridge=1e-5*m, LM damping=1e-3*m, LM scale floor=1e-6*m,
trust radius=.1/m, pairwise cap=2/m, coordinate cap=16/m. This extends the
measured protection curve toward the unprotected limit, not toward a primary
posterior metric. Tiny positive guards may be ineffective at FP32 resolution:
that is a hypothesis to test, not evidence that a protection is unnecessary.
Nominal health, flagged stress failures, no new healthy-to-invalid stress row,
and actual common-randomness HMC proposal materiality remain mandatory. Stress
at three prior SD is diagnostic; an unflagged failure is a continuation veto.

Proposals use the previously recorded .05/.2 normalized steps, L2, four starts,
and two stateless streams. Each arm preserves all 16 proposal probabilities,
health/status traces, and filter diagnostics. The independently frozen endpoint
is the same within a matched comparison. The heuristic adversary here is the
adjacent weaker protection under identical data/momenta; posterior baseline
comparisons remain required later. These short probes cannot establish mixing,
posterior correctness, force-gradient equality, or speed superiority.

Skeptical audit: the preceding failed pair is not retested on the same holdout,
its .01 tolerance is not relaxed, and force health cannot be overruled by small
probability changes. A common passing pair establishes only scoped control
selection. If none passes, preserve that result and localize the non-monotone
force/proposal sensitivity within the remaining repair allocation. Do not launch
unselected HMC. Corrupt artifacts, wrong scope/data, or unflagged invalidity stop
this worker; a flagged candidate failure motivates repair.

## Compute, artifacts and command

This repair reserves4000 CPU-worker and4000 GPU-wall seconds from the approved
10000/10000 repair allocation. This leaves6000/6000 for other repairs; the total
continuation144000CPU/126000GPU and global198000CPU/226800GPU caps still apply.
The worker cap is3800/3800; focused CPU-only review120 seconds. No source changes
while this master pass runs. Each retry uses a fresh versioned directory.

```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-materiality-repair --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-materiality-repair-01
```

Run with escalated GPU access, verified growth, TF32 enabled and XLA; preserve
run manifests, source hashes, logs and structured reports. Implementation review
covers original and repair issuance/reload, incorrect fresh-data rejection,
unchanged tolerance, native HMC call chain and total budget enforcement.

| Decision | Primary criterion | Vetoes | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Run bounded repair | Common .01 pair on fresh partitions | Numerical health and evidence validity | Probes cover finitely many trajectories | Freeze only if both pass, then public tuning | Posterior correctness |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Enforced by actual status and saved-proposal checks |
| Statistically supported ranking | None sought |
| Descriptive differences | Finite probe acceptance differences and runtime |
| Default readiness | No promotion; existing FP32 default retained |
| Next evidence | Public fresh tuning verification and retained/reference comparisons |
