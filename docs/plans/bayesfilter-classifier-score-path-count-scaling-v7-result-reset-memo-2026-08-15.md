# V7 Path-Count Scaling Result Reset Memo

Date: 2026-08-15  
Status: `BLOCKED_BY_SIR_GENERATIVE_NONFINITE_PATH`

## User GPU Policy

GPU execution remains the repository default. The live policy is now explicit:

- prefer physical `nvidia-smi` GPU 1;
- an eligible GPU has utilization `<50%` and free memory `>8192 MiB`;
- fall back to physical GPU 0 only when GPU 1 is unavailable or ineligible;
- bind the selected physical device by UUID because CUDA ordinal `1` did not
  map to physical `nvidia-smi` GPU 1 on this host; and
- record physical index, UUID, name, launch utilization, free memory, and
  selection reason in the run manifest.

At the last trusted probe, both devices qualified:

| Physical index | Device | Utilization | Free memory | Decision |
|---:|---|---:|---:|---|
| 1 | RTX 4080 SUPER | 0% | 16,035 MiB | selected |
| 0 | RTX 5080 | 8% | 11,842 MiB | fallback |

The shared policy and live selector were verified by focused tests and a
trusted selector call. TensorFlow UUID binding was separately verified to
resolve physical GPU 1 to the RTX 4080 SUPER.

## Approval Diagnosis

The earlier approval problem was an automatic permission-review timeout before
process creation. It was not a scientific veto and did not establish a CUDA,
TensorFlow, XLA, memory-growth, or GPU-health failure. A later launch did start,
but the old wrapper's `CUDA_VISIBLE_DEVICES=1` selected the RTX 5080 because
CUDA ordinal ordering differed from `nvidia-smi` index ordering. That attempt is
preserved as invalid infrastructure evidence and is not included in any result.

## Executed Diagnostics

| Attempt | Artifact | Outcome | Role |
|---|---|---|---|
| wrong-device capacity launch | `capacity_sir_16384_status/worker.log` | TensorFlow used RTX 5080; fit then failed non-finite | invalid infrastructure attempt |
| exact V7 SIR bank scan, provenance-complete corrected trace | `sir_16384_nonfinite_diagnostic_attempt03.json` | one non-finite path | hard validity veto |

The exact V7 seeds, bundle 0, coordinate `j=1`, and all six perturbations were
scanned at `N=16,384` in the frozen 8,192-row blocks. There was exactly one
non-finite trajectory among the plus/minus training banks:

- `delta=0.01`, plus class;
- global path index `9864` in the second 8,192-row block;
- first non-finite observation at `t=11`.

The corrected scalar trace reproduces the same implemented Zhao-Cui RK4
variant outside the large batch. At `t=10`, the state reaches approximately
`1.1e4`; at `t=11`, the RK4 substeps grow through approximately `9.2e10` and
`7.6e120`, then become non-finite. This is not caused by GPU memory pressure,
XLA batch shape, classifier optimization, or the approval boundary.

The detailed component trace is preserved in
`sir_failure_trace_attempt01.json`; the standalone-versus-canonical lockstep
comparison is preserved in `sir_canonical_path_comparison_attempt01.json`.
The latter reports zero transition-mean and state differences at every finite
step `t=1..10`, followed by the canonical error `state: NONFINITE_VALUE` at
`t=11`.

The integrator-sensitivity diagnostic is preserved in
`sir_integrator_sensitivity_attempt01.json`. It replays the same noise path
with both the author half-step RK4 and classical RK4, using internal steps from
`0.005` down to `0.0001`. Refinement does not restore a finite path: both
variants fail around the same horizon, and the smaller steps fail no later than
the baseline. This rules out ordinary roundoff and the author fourth-stage
convention as the primary cause; the refined integrators follow the same
unstable negative-infectious dynamics more faithfully.

The archived author Live Script source confirms the policy boundary directly:
`st_process.mlx` computes `sir_step(...) + sigma1*randn(...)` and applies
`max(...,0)` only to odd MATLAB coordinates (susceptible states). It contains
no infectious clipping, positive-state transform, or finite-state rejection.

The source audit confirms that the standalone V7 simulator matches the
repository's declared source variant: half-step fourth RK stage and
susceptible-only clipping after additive process noise. A lockstep replay of
the exact path against `ParameterizedZhaoCuiSIRSSM` agrees at both transition
mean and post-noise state through every finite step (`t=1..10`); both routes
reject/become non-finite at `t=11`. This is not a V7 transcription mismatch or
GPU/XLA numerical discrepancy.

The root cause is a model-domain contract defect for the intended SIR
semantics. The source-faithful push clips only susceptible coordinates, while
Gaussian process noise can make infectious coordinates negative. In the
implemented RHS,

`dI_j = I_j * (kappa_j*S_j - nu_j) + diffusion_j`.

For the failing `theta=[0,0.01,0]` path, `nu/kappa ~= 181.8`. The first
transition noise makes compartment 9 `I9=-1.577`; by `t=8` the path is
`S9=650.1, I9=-109.7`, then `S9=923.4, I9=-322.0` at `t=9`, and
`S9=11038.1, I9=-9940.9` at `t=10`. Since `S9 > nu/kappa` while `I9 < 0`,
the local infection/recovery term amplifies the negative infectious state.
The next RK4 stages reach approximately `9.2e10` and `7.6e120` before
becoming non-finite. The hard failure is therefore a deterministic instability
of an unphysical negative-infectious state, triggered by an otherwise ordinary
Gaussian process-noise draw. It is not merely floating-point roundoff.

The code-level mismatch is between the physical name/domain of the SIR model
and its declared source push policy: `domain_policy="diagnose_negative_after_noise"`
is recorded by the canonical model, but the active author route does not
reject or repair negative infectious coordinates. The V7 standalone simulator
also has no state-domain check; it only discovers the failure when observations
become non-finite. Changing this requires a reviewed target-law decision
(infectious clipping, a positive-state transform, bounded/non-Gaussian process
noise, or a stable integration/domain rejection policy). Any such repair
changes the simulation law and needs fresh nested baselines; silently applying
one would invalidate the current comparison.

## Research-Question Guardian

| Ledger field | Current status |
|---|---|
| Main question | Does independent classifier training variance decrease from 8,192 to 16,384 and follow `1/N`? | not testable for SIR at this stage |
| Baseline | V6 `independent_n8192` | preserved |
| Promotion criterion | paired audit variance ratio and `1/N` interval | not reached |
| Hard veto | all generated training observations finite | failed at one exact nested path |
| Repair trigger | review/repair SIR finite generative program without silently censoring paths | active |
| Nonclaim | no exact SIR score, no filter validation, no `1/N` conclusion | remains in force |

## Why No Rows Were Dropped

Dropping, clipping, replacing, or resampling the one failing row would change
the simulated observation law and break the exact nested-prefix comparison.
The current V7 experiment therefore stops rather than silently conditioning on
numerical survival. The 16,384 and 32,768 classifier campaigns must not be
launched under the current simulator.

## Opt-In Survivor-Conditioned Diagnostic (2026-08-16)

At the user's request, the runner now has an explicit `remove_invalid_paths`
arm. It is not the canonical V7 law: invalid paths are removed as matched
`+/-` pairs to preserve classifier class balance, held-out splits and the fixed
observation remain unchanged, and the result is labeled
`survivor_conditioned_training_law`. The alert threshold is `0.1%` (`1e-3`)
and is reported independently from the opt-in removal policy.

The bounded run
`docs/benchmarks/artifacts/classifier_score_path_count_scaling_20260815/sir_16384_filtered_attempt01/`
used bundle 0, `N=16,384`, `T=50`, coordinate `j=1`, GPU 1 (RTX 4080 SUPER),
and completed in 310.0 seconds. It generated 196,608 training rows, found one
invalid row (`5.086e-6`), removed one matched pair (two classifier rows), and
therefore did not cross the `0.1%` alert threshold. The filtered fit was finite
and optimizer-complete after 15 epochs, with 194,560 retained training rows.

This is a sensitivity diagnostic only. Its classifier estimates the law
conditional on finite-path survival; it does not recover the original SIR
likelihood ratio or validate the SIR generative model. No path-count scaling
claim is made from this single filtered cell.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| V7 SIR 16,384 stage | no valid fitted estimator exists | failed | whether the target law should be made finite or the integrator made stable | define and review a source-preserving numerical repair, then rerun fresh nested artifacts | no claim about path-count scaling or classifier quality |
| GPU policy | physical-device selection and provenance | passed | live utilization can change after preflight | selector rechecks at every launch | no claim about scientific validity |
| Approval boundary | trusted process creation | resolved for this run | automatic reviewer may still time out on future compound commands | use narrow trusted commands and persistent status wrappers | no claim that approval is scientific evidence |

## Post-Run Red Team

The strongest alternative explanation is that the author-source SIR law itself
permits rare explosive trajectories because additive Gaussian process noise is
unbounded and the infection term is quadratic. The trace diagnostic makes this
explanation consistent with the evidence, but it does not decide whether the
research target should instead use a smaller stable integration step, a bounded
noise law, a different state transform, or an explicitly declared finite-path
domain. Any such change alters the generative target and requires a new reviewed
plan and fresh baseline.

The result does not invalidate the classifier method, the V6 8,192 artifacts,
or GPU setup. It invalidates only the current SIR V7 continuation under the
unchanged simulator because the required finite-data invariant fails.
