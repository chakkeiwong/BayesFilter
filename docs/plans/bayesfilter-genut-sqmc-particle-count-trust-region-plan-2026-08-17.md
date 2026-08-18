# GenUT SQMC Particle-Count and Trust-Region Comparison Plan

Date: 2026-08-17  
Status: `COMPLETED_DIAGNOSTIC`

## Research Intent Ledger

| Field | Frozen decision |
|---|---|
| Main question | In the existing Full-SQMC LEDH-PFPF-GenUT test set, does increasing the particle count from the prior baseline reduce replicate value/score variance, and does the repaired GenUT dual-cap trust-region reset improve finite validity or variance relative to the legacy dual-cap reset? |
| SQMC scope | The saved full-horizon four-route campaign. The first discriminating execution is Austria-SIR `T=20`, where the unresolved `j0` variance is the stated hypothesis. The recovered six-model `T=6,N=72` campaign is a port smoke only. |
| Baseline | `iid_dual_cap`, `previous_inverse_cdf`, `repaired_fixed_previous_controls`, and `repaired_permutation`, with their recovered fixed controls and map calibration. |
| Candidate | Full-SQMC with the repository route ID `genut_column_scaled_lm_smooth_rms_trust_v1` applied to the diagonal higher-moment GenUT correction inside the Contract-E reset. The candidate is explicitly experimental; the analytical all-parent filtering score remains unchanged. |
| Particle counts | `N=1008`, `N=2016`, and `N=4032`, corresponding to the prior full-horizon baseline and the requested 2x/4x tests. All are divisible by the Austria `2d=36` design period. |
| Replications | Independent randomized-Halton seeds `97701..97716`, paired across routes, reset variants, and particle counts. Within one randomized point set, rows are not treated as IID replications. |
| Primary diagnostic | Per model and arm: replicate variance of value and total score, plus the scaling products `N * Var(value)` and `N * Var(score component)`; report SD ratios against the expected `1/sqrt(2)` and `1/2` landmarks. |
| Trust comparison | At `N=1008`, compare legacy dual-cap and repaired trust-region dual-cap with identical observations, seeds, point sets, event order, and non-reset controls. If all legacy smoke cells are invalid, preserve that veto and use the larger-count budget for trust-region particle scaling only. |
| Promotion criterion | None. This is a bounded diagnostic campaign only. A lower variance or higher finite rate nominates a follow-up; it does not establish superiority, unbiasedness, exact filtering, or default readiness. |
| Hard vetoes | Wrong target/event order/hash, missing SQMC arm, nonfinite value/score, invalid reset, state-map saturation, invalid full-state Hilbert ordering, score recursion mismatch, stale/mismatched trust controls, or artifact corruption. |
| Explanatory diagnostics | ESS, maximum normalized weight, unique ancestors, Hilbert ties, reset residuals, covariance gap, LM condition, pre/post trust RMS, cap activity, runtime, and allocator usage. |
| Nonclaims | No formal SQMC theorem transfer through Contract-E, no exact nonlinear likelihood/score, no statistical ranking from 16 seeds, no claim that the trust cap is unbiased, no HMC/NeuTra/default readiness, and no causal attribution to ancestry versus innovation ordering. |

## Trust-Region Route Contract

The candidate controls are frozen from the repaired result note:

```text
higher_moment_lm_damping = 1e-2
higher_moment_lm_scale_floor = 1e-4
higher_moment_trust_radius = 0.5
```

The route identity is `genut_column_scaled_lm_smooth_rms_trust_v1`. The repair
uses column-scaled Levenberg--Marquardt coefficients and a smooth row-RMS cap;
it must not clip the target moments or silently fall back to the legacy solve.
The SQMC lane's standard score is the repository analytical all-parent
backward filtering score, not a derivative of the finite SQMC/reset program.
Therefore this campaign validates the repaired reset value path and score
finiteness/variance separately; it does not claim the trust-region JVP gate for
the analytical score.

## Evidence Contract

| Role | Requirement |
|---|---|
| Engineering | Ported SQMC runner and trust-reset adapter import cleanly; Hilbert ordering, inverse-CDF ancestry, point-set hashes, and particle shapes pass focused tests; XLA graph has static shapes and no scalar fallback. |
| Numerical | Every required row is finite, `program_valid`, unsaturated, has valid reset/covariance diagnostics, and uses the declared route identity. Invalid rows are retained and reported, never discarded. |
| Particle scaling | For each Austria-SIR arm, compare `N=1008,2016,4032` using paired seeds. `N Var` stability is descriptive evidence for approximately `1/N` behavior; it is not a theorem or promotion gate. |
| Trust effect | Compare legacy and trust-region validity at `N=1008`. If every legacy route is invalid there, retain that hard-veto evidence and do not spend the larger-count budget on an already-vetoed legacy route. Interpret finite variance differences descriptively. |
| Artifact | Fresh versioned output root containing raw rows, summaries, manifest, source hashes, and a result/reset memo with decision and inference-status tables. |

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|
| Six-model SQMC set | Existing full-SQMC result/reset memo | current main omitted this experimental lane | exact source-path and target-hash ledger | ported opt-in scope |
| `N=1008,2016,4032` | prior full-horizon baseline plus 2x/4x request, preserving divisibility by the Austria `2d=36` design period | larger N may exceed GPU/XLA memory/time | smoke at each N; allocator/runtime record | diagnostic ladder |
| 16 randomized seeds | prior SQMC pilot | low power for ranking and tails | variance/MCSE and explicit nonclaim | descriptive replication |
| Halton rather than Sobol | existing SQMC implementation identity | different net properties and no theorem transfer | point-set ID/hash in manifest | inherited implementation |
| Trust controls | 2026-08-15 repaired result | route may be incompatible with old primal reset or change finite objective | route-ID check and legacy/candidate same-target smoke | opt-in hypothesis |
| CPU Host-XLA first | saved SQMC mechanics evidence and safe port check | differs from GPU production target | `CUDA_VISIBLE_DEVICES=-1`, `_XlaMustCompile`, finite smoke | reference lane |
| GPU follow-up | repository default GPU and user requested execution | memory/compile failure at `N=2016` or `N=4032` | trusted GPU probe, growth verification, allocator record | only after CPU mechanics gate |

## Skeptical Plan Audit

1. **Wrong baseline:** the ordinary four-model IID GenUT harness is not called
   SQMC. The plan uses the saved six-model Full-SQMC runner and preserves its
   IID and initial-only arms.
2. **Route mismatch:** the saved SQMC lane currently uses a primal legacy
   dual-cap reset. A run is not accepted as a trust-region test unless the
   candidate control reaches the repaired LM/smooth-cap implementation and the
   artifact records the route ID. If the required adapter cannot be made
   finite and shape-valid, the campaign stops as an implementation blocker.
3. **Unfair particle comparison:** seeds, observations, point-set construction,
   event order, reset design, and all non-reset controls are held fixed within
   each paired comparison. Counts are separate static XLA graphs.
4. **Proxy promotion:** variance, ESS, residuals, and runtime are not promoted
   to correctness or superiority criteria. The exact LGSSM reference remains
   explanatory only in this first ladder.
5. **Silent invalid-path deletion:** any invalid row vetoes that cell and is
   preserved in raw output; it is not removed to improve summaries.
6. **Stale tuning:** old SQMC/dual-cap settings are warm starts only. The trust
   controls are frozen by the repaired-result artifact, and no claim-data
   tuning is performed in this bounded diagnostic.
7. **Resource stop:** stop before a larger count if compile/runtime or allocator
   diagnostics exceed the declared budget; do not silently lower N and label it
   the requested test.

Audit decision: `PASS_WITH_ROUTE-PORT_BLOCKER_EXPLICIT`. Execution may begin
with the port and focused smoke. Scientific comparison begins only after the
trust-route identity and SQMC mechanics gates pass.

## Execution Ladder And Budget

1. Port the minimum SQMC modules/runner and add a reset control that selects
   `legacy_dual_cap` or `genut_column_scaled_lm_smooth_rms_trust_v1` without
   changing global defaults.
2. Add focused CPU-hidden tests for route identity, trust controls, static
   shapes, point-set pairing, and invalid-row preservation.
3. Run the recovered six-model `T=6,N=72` mechanics smoke, then one Austria
   `T=20` seed at `N=1008` for all four routes and both reset variants.
4. Run the paired legacy/trust comparison at Austria `N=1008`. If the legacy
   reset is hard-vetoed there, run the trust-region particle ladder at
   `N=1008,2016,4032`. Execute `N=2016` before `N=4032`; do not launch 4x if
   the 2x cell is invalid or exceeds the resource stop. The other models are
   not used to answer the Austria `j0` particle-count hypothesis; they remain
   future cross-model validation.
5. Run a GPU/XLA replay only if the CPU port is valid and GPU1 has less than
   50% utilization and more than 8 GiB free. Set memory growth before import;
   prefer physical GPU1 (`CUDA_VISIBLE_DEVICES=0` on this host), otherwise GPU0.
6. Write the result/reset memo with particle-scaling tables, paired trust
   tables, hard-veto table, inference-status table, and post-run red-team note.

Campaign ceiling: 30 CPU minutes for port/smoke, 180 GPU minutes for the
Austria ladder, two localized infrastructure retries, and no default/API/HMC
changes. Every attempt uses a fresh output directory and records the actual
command, environment, commit, dirty-path hashes, seeds, device, wall time, and
artifact paths.

## Planned Artifacts

`docs/benchmarks/artifacts/genut-sqmc-particle-count-trust-region-20260817/`

Terminal note:
`docs/plans/bayesfilter-genut-sqmc-particle-count-trust-region-result-2026-08-17.md`

## Post-Execution Documentation Correction

The launch-time plan contained stale `N=72,144,288` text in two audit-table
rows even though the research ledger, execution ladder, runner constants, and
actual commands consistently specified `N=1008,2016,4032`. The terminal claim
artifact preserves the launch-time plan SHA-256. This post-run correction
aligns the prose with the executed scope and records the predeclared stopping
decision used after all four legacy `N=1008` smoke cells failed.
