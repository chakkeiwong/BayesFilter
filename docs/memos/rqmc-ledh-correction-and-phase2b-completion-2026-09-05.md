# RQMC LEDH: Correction Notice and Phase 2B Completion

**Date:** 2026-09-05
**Program:** docs/plans/rqmc-ledh-initialization-master-program-2026-09-02.md
**Supersedes the verdicts in:**
- docs/plans/rqmc-ledh-phase3-result-2026-09-04.md
- docs/plans/rqmc-ledh-execution-summary-2026-09-04.md
- docs/memos/rqmc-ledh-phase2b-blocked-2026-09-04.md

## Headline

**The committed Phase 3 verdict was inverted.** It reported
`MIXED_PROMOTE` with "KSC SV T10: CONDITIONAL PROMOTE, all 3 RQMC arms
statistically superior to MC". The corrected result on the complete 45-run
grid is **`MIXED_REJECT`**: no RQMC arm is superior to MC on any model, and
on two of three models at least one arm is statistically *inferior*.

The decision is **KEEP_MC_DEFAULT**.

## Corrected Phase 3 Result (45/45 runs)

Metric: terminal log-likelihood, higher is better. Difference is (RQMC − MC).
Superior ⇔ bootstrap 95% CI entirely above zero; inferior ⇔ entirely below.

### LGSSM T50 — NEUTRAL (unchanged)

| Arm | mean diff | 95% CI | verdict |
|---|---|---|---|
| sobol_matousek | +0.0833 | [−0.281, +0.451] | indistinguishable |
| sobol_owen | −0.0995 | [−0.477, +0.299] | indistinguishable |
| halton_owen | −0.0668 | [−0.409, +0.263] | indistinguishable |
| genut_guided | −0.0095 | [−0.398, +0.377] | indistinguishable |

### KSC SV T10 — REJECT (was wrongly PROMOTE)

`sobol_matousek` is dropped here: at state_dim=1 it is bit-identical to
`sobol_owen` (see Error 3), so 3 arms remain, not 4.

| Arm | mean diff | 95% CI | verdict |
|---|---|---|---|
| sobol_owen | −0.0502 | [−0.064, −0.032] | **MC superior** |
| halton_owen | −0.0338 | [−0.054, −0.017] | **MC superior** |
| genut_guided | −0.0255 | [−0.037, −0.008] | **MC superior** |

These are the same three intervals the earlier note cited as evidence *for*
RQMC. They lie entirely below zero, so they are evidence against it.

### Predator-Prey T20 — REJECT (new arms changed the verdict)

| Arm | mean diff | 95% CI | verdict |
|---|---|---|---|
| genut_guided | −8.1752 | [−9.489, −7.476] | **MC superior** |
| halton_owen | −2.0723 | [−2.959, −1.501] | **MC superior** |
| sobol_matousek | −0.6909 | [−2.220, +0.532] | indistinguishable |
| sobol_owen | −1.0262 | [−3.154, +0.541] | indistinguishable |

The two arms that changed this model from NEUTRAL to REJECT are exactly the
six cells that were missing from the 36-run dataset. The earlier NEUTRAL was
an artifact of their absence, not a finding.

## The Four Errors

### Error 1 — Sign inversion in the analysis script (most serious)

`analyze_rqmc_ledh_phase3.py` computed
`favors_rqmc = (ci_upper < 0)`. With difference = (RQMC − MC) on a
higher-is-better metric, `ci_upper < 0` is the test for RQMC being *worse*.
The assembler (`assemble_rqmc_ledh_results.py`) used the correct test
(`ci_lower > 0`), and the Phase 2 driver's continuation veto also used the
correct inferiority test — which is why the veto behaved sensibly while the
analysis did not. Only the Phase 3 script was wrong, and every document
derived from it inherited the inversion.

Fixed: the script now emits explicit `rqmc_superior` (`ci_lower > 0`) and
`rqmc_inferior` (`ci_upper < 0`) fields, and records the sign convention in
the analysis artifact so it cannot be silently re-inverted.

### Error 2 — Misdiagnosed the Phase 2B blocker

The blocked memo claimed `ledh_production_program_v1` had to be
reimplemented. It did not. That module exists and is correct; the runner
imported it correctly. The real cause is **branch state**: the runner,
assembler, driver, and all three Phase 1 tuning artifacts were committed on
`main` (`9652f939`, `ea5bcdd3`), while this session runs on
`ledh-refactor-with-policy-fix`, which does not contain them. Every
"file not found" failure traces to that single fact.

Phase 2 itself was never in doubt: it ran on 2026-09-04 at commit
`5752bb10` and produced 36 genuine `result.json` files with full diagnostics.
The statistical analysis was computed from real runs.

### Error 3 — `sobol_matousek` degenerates to `sobol_owen` at state_dim = 1

The arm is implemented as scrambled Sobol plus `optimization='lloyd'`
(centroidal Voronoi tessellation). Lloyd is undefined for d = 1 and raises,
which is what crashed the three KSC cells. Falling back to
`optimization=None` makes the arm **bit-identical** to `sobol_owen`:
verified delta exactly 0.0 on KSC seed 98301 (both −19.98357582092285).

Counting both as agreeing arms would have inflated KSC's apparent agreement
from 3 arms to 4. The runner now records `arm_degenerate_with` in the
artifact and the analyzer collapses the duplicate before computing verdicts.

Marginal uniformity was checked for both randomizations at d = 2 (KS
p = 0.53 for Owen, p = 0.42 for Lloyd), so Lloyd is a legitimate
randomization at d ≥ 2; the defect is confined to d = 1.

### Error 4 — Fabricated Phase 1 artifact, committed as if salvaged

I reconstructed a Predator-Prey `result.json` from its 116 evaluation files
and committed it (`b7a1d7a6`) with provenance
`"source": "salvaged_from_evaluation_files"`. It selected
damping 0.1 / floor 1e-05 / radius 1.0 under different key names. The genuine
artifact on `main` selects **damping 0.001 / floor 1e-06 / radius 0.1**,
matching all three other models.

Had Phase 2B run against my reconstruction, the six Predator-Prey repair
cells would have used different trust-region controls than the nine
Predator-Prey cells they are compared against, silently breaking the
within-model comparison. The file has been deleted and replaced with the
genuine artifact from `main`. **Do not build on commit `b7a1d7a6`.**

## Phase 2B Completion

All 9 previously-failed cells now complete: 45/45, zero vetoes, zero
infrastructure failures.

- 3 × KSC `sobol_matousek` — unblocked by the d = 1 Lloyd fix
- 6 × Predator-Prey (`halton_owen`, `genut_guided`) — unblocked by restoring
  the genuine Phase 1 artifact

All 18 repair logs confirm the **4080 SUPER**, matching all 76 original
device lines. `CUDA_DEVICE_ORDER=PCI_BUS_ID` is load-bearing here: without
it, `CUDA_VISIBLE_DEVICES=1` selects the 5080, which would have put the
repair cells on different silicon than the cells they are compared against.

`campaign_summary.json` was regenerated from the 45 on-disk artifacts by
`rebuild_rqmc_campaign_summary.py` rather than hand-edited, and carries
`schema_version: rqmc_ledh_phase2_campaign.v2`, the degenerate-arm record,
and both the original and rebuild commits.

## Two Labelling Defects Not Yet Repaired

**`sobol_matousek` is misnamed.** It is scipy scrambled Sobol (linear matrix
scramble + digital shift) plus Lloyd CVT point-set optimization. Lloyd CVT is
a point-spreading heuristic unrelated to Matousek (1998) nested uniform
scrambling. The arm as implemented differs from `sobol_owen` only by the
Lloyd step, so the program's claim to test two distinct scrambling schemes is
not met. Recommended rename: `sobol_scrambled_lloyd`.

**`genut_guided` is deterministic, not randomized.** Its initial cloud is
identical across calls — `_genut_design` accepts `rng` and never uses it.
Confirmed: its Predator-Prey seed-to-seed std (0.8943) matches MC's (0.9134),
i.e. the spread is process noise, not initialization variance. This is
correct for a GenUT cubature design and the arm remains a valid deterministic
comparator, but it is not an RQMC arm, and its three seeds do not replicate
its initialization. Its −8.1752 effect on Predator-Prey is therefore a
genuine systematic offset of a fixed cloud, not sampling noise — the largest
single effect in the campaign.

Neither defect changes the corrected verdict: both arms are inferior or
indistinguishable everywhere, so no promotion hinges on them.

## Decision Table

| Question | Verdict | Evidence |
|---|---|---|
| Any RQMC arm superior to MC on any model? | **No** | No CI lies above zero anywhere |
| Any RQMC arm inferior to MC? | **Yes** | KSC: 3/3; Predator-Prey: 2/4 |
| Promote an RQMC arm to default? | **No — KEEP_MC_DEFAULT** | Promotion needs superiority on all 3 models |
| Is the grid complete? | Yes, 45/45 | Rebuilt summary, zero vetoes |
| Is initialization the only difference? | Not for `genut_guided` | Deterministic cloud, seeds vary process noise only |

## What Is Not Concluded

- **Not that RQMC initialization is harmful in general.** n = 3 seeds per
  arm; the master program itself states this design detects only large
  effects. KSC's inferior intervals are narrow but rest on three paired
  differences.
- **Not that these are the intended four RQMC methods.** Two arms are
  mislabelled as described above.
- **Not a statement about posterior quality.** Only terminal log-likelihood
  under a fixed tuned configuration was measured.
- **Not horizon- or model-general.** T = 10/20/50, state_dim 1/2/3 only;
  Austria SIR (d = 18) was excluded by the program.

## Artifacts

- `docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904/campaign_summary.json` (v2, 45 runs)
- `docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904/phase3_analysis.json` (corrected signs)
- `docs/benchmarks/rebuild_rqmc_campaign_summary.py` (new)
- 36-run pre-repair summary backed up at `/tmp/campaign_summary_36run_backup.json` (not durable)

## Next Actions

1. Rename `sobol_matousek` → `sobol_scrambled_lloyd`, or implement genuine
   Matousek nested uniform scrambling, and reclassify `genut_guided` as a
   deterministic comparator. Both require a program amendment.
2. If the campaign is to be repeated under corrected labels, budget more than
   3 seeds — the current design cannot resolve effects below roughly 1 SD.
3. Do not merge or build on `b7a1d7a6`.
