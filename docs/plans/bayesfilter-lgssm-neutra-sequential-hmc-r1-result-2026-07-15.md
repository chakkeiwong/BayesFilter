# LGSSM NeuTra Sequential HMC Repair Phase R1 Result

Date: 2026-07-15  
Decision: `PASS_R1_BOTH_FIXED_KERNELS_ADMITTED`

## Outcome

Both independently trained frozen NeuTra candidates passed the corrected fresh
sequential warm-up and tuning-admission controller. Each used four chains in
one CPU-hidden TensorFlow/TFP XLA batch, float64, step size `0.8`, and 10
leapfrog steps. Every warm-up and retained chunk was separately archived in
latent and raw coordinates; warm-up was not included in posterior draws.

| Candidate | Warm-up draws/chain | Warm-up max modern R-hat | Retained at 1,000 | Final retained draws/chain | Final max modern R-hat | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense_seed1201` | 2,000 | `1.010715` | fail: `1.010823` | 2,000 | `1.006396` | `560.91 s` |
| `dense_seed1202` | 2,000 | `1.010573` | fail: `1.011698` | 2,000 | `1.005702` | `564.30 s` |

Warm-up readiness used the latest 1,000 archived warm-up transitions and the
predeclared threshold `<=1.05`. Retained admission used cumulative draws and
the stricter threshold `<=1.01`. All modern R-hat rows were finite. Every chunk
had finite states, target values, and log acceptance; valid target-status
telemetry; all-chain movement; and zero energy-error divergence screens.

The two initial 1,000-retained checks are the decisive repair evidence: both
missed the gate and correctly triggered another 1,000 draws rather than a
candidate rejection. Both then passed at 2,000. This establishes that the
fixed 1,000-draw terminal decision was a planning error for these runs.

## Artifacts

Aggregate result:
`docs/plans/artifacts/lgssm-neutra-gap-closure-2026-07-15/sequential-repair-attempt-01/result.json`,
artifact hash
`sha256:a67aefd6f7a8b03494f2635fa8ab6498cec24022c8b3e05e91ffd46b1c1cf484`.

Candidate result hashes:

- seed1201:
  `sha256:14a75b67791aa6d58b1af933877e3c2473075bb1491ddc6e08bb641b10d92abf`;
- seed1202:
  `sha256:db7e91fe7d5a195f543eff1be11138a4747d32899b27462adb892d5114629c27`.

Each candidate has two warm-up chunks, two retained chunks, and verified
cumulative TensorFlow archives for latent and raw coordinates. Historical
Phase 4 outputs were not overwritten.

## Decision And Inference Status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit both fixed kernels to fresh confirmation | cumulative raw-coordinate modern R-hat `<=1.01` after sequential warm-up | no health, status, identity, artifact, divergence, or cap veto | confirmatory ESS, posterior agreement, and truth recovery | run R2 with fresh seeds and 4,000 minimum / 10,000 cap | no posterior correctness, recovery, superiority, or default claim |

| Inference status | Verdict |
| --- | --- |
| Hard veto screen | both candidates pass |
| Statistically supported ranking | none; candidates are not ranked |
| Descriptive-only differences | observed R-hat, acceptance, and timing values |
| Default readiness | not evaluated |
| Next evidence needed | fresh confirmatory full convergence, comparator agreement, and truth-recovery checks |

## Post-Run Red Team

The strongest alternative explanation is that the fresh 2,000-draw admissions
are favorable stochastic screens that will not persist under fresh seeds or
will have inadequate ESS/posterior agreement. R2 directly tests that. A failed
R2 candidate will reject that candidate under the confirmatory contract, not
invalidate the target, training harness, or NeuTra research direction.

## Handoff

Continue to
`docs/plans/bayesfilter-lgssm-neutra-sequential-hmc-r2-subplan-2026-07-15.md`.
Both original confirmation seeds remain unused. Aggregate R1 compute was about
18.8 minutes, leaving ample room within the six-hour campaign ceiling.
