# HNN-NeuTra Exact-Gradient Comparison Phase 4 Result

Superseded for tuning and performance claims, 2026-07-18.  The fixed-grid
tuner used here was not the native BayesFilter dual-averaging tuner.  Tuned
runtime, seconds/ESS, speed ratio, break-even, and performance decisions below
are `UNSUPPORTED_PENDING_NATIVE_RETUNING`; the historical values remain only
as provenance.  See
`bayesfilter-hnn-neutra-native-tuning-correction-result-2026-07-18.md`.

Decision: `PASS_SIR_FULL_COMPARISON_STR_HNN_VALID_EXACT_BASELINE_UNRESOLVED`.

## Outcome

SIR-SGQF passed the complete one-seed, same-chart comparison between
HNN-NeuTra-HMC and exact-gradient NeuTra-HMC. STR-UKF's HNN arm independently
passed sampler validity, generating-truth, and deterministic structural gates,
but its exact-gradient comparator had repeated rare energy catastrophes before
retained sampling. STR-UKF therefore has no direct HNN-versus-exact posterior
agreement result.

| Cell | HNN accuracy | Exact accuracy | Direct agreement | Matched mechanics | Decision |
| --- | --- | --- | --- | --- | --- |
| SIR-SGQF | pass | pass | pass, max `z_MC=0.1281` | pass, `25.564x` descriptive ratio | full one-seed pass |
| STR-UKF | pass | unresolved | unavailable | pass after repair, `14.031x` descriptive ratio | HNN viable; full comparison unresolved |

## SIR-SGQF

Both arms selected `epsilon=0.8`, `L=10`, retained 1,000 draws per chain after
2,000 warm-up draws per chain, and passed finite-energy, endpoint-count,
rank-normalized split/folded R-hat, bulk ESS, tail ESS, truth-tail, and direct
posterior-agreement gates.

| Metric | HNN | Exact gradient |
| --- | ---: | ---: |
| Acceptance | 0.8968 | 0.8973 |
| Max retained R-hat | 1.00363 | 1.00337 |
| Min bulk ESS | 5089.6 | 5076.4 |
| Min truth tail | 0.3737 | 0.3742 |
| Tuned sampling | 40.883 s | 1034.786 s |
| Seconds/min bulk ESS | 0.008033 | 0.203845 |

The matched benchmark held chart, initial positions, seed, endpoint, four
chains, 500 transitions per chain, `epsilon=0.8`, and `L=10` fixed. Synchronized
warm medians were 6.773 seconds for HNN and 173.135 seconds for exact gradient,
a descriptive `25.564x` ratio. All physical-coordinate intervals overlapped;
the maximum pooled-MCSE discrepancy was `0.1281`. No second seed was required.

Supervision generation took 100.841 seconds and the HNN training grid took
9.664 seconds. HNN preparation therefore amortizes after 333 matched transition
batches. The complete independently tuned reuse campaign has immediate
break-even because exact-gradient tuning alone took 1253.855 seconds versus
84.145 seconds for HNN tuning. These are different break-even questions.

## STR-UKF

The HNN arm selected `epsilon=0.2`, `L=12`. It retained 4,000 draws per chain
after 2,000 warm-up draws per chain and passed all five physical truth tails.
Its maximum retained R-hat was `1.00963`, minimum bulk ESS was `5884.5`, minimum
tail ESS was `1570.6`, and minimum truth tail was `0.2885`. The executed filter
manifest preserved
`k_t=phi*k_(t-1)+gamma*m_t^2` and
`artificial_k_noise_allowed=false`.

The independently tuned exact arm selected the boundary candidate
`epsilon=0.2`, `L=8`. Its 500-transition tuning probe passed the predeclared
health threshold with maximum `|delta H|=85.39`, but its first 1,000-transition
warm-up chunk produced maximum `|delta H|=4213.19`. No retained draws were
taken.

The one allowed localized repair used the already measured safer in-grid
candidate `epsilon=0.1`, `L=8`, with the same target, chart, data, seeds,
initialization, hardware, and thresholds. Its tuning probe had maximum
`|delta H|=16.05`, but the serious warm-up still produced a rare
`|delta H|=1307.24`. The repeated exact-baseline health failure stops further
STR-UKF retries under this runbook. It is an exact-HMC tuning/numerical
instability on this chart, not an HNN or structural-model failure.

A separate repaired matched-mechanics benchmark at `epsilon=0.1`, `L=8`
passed its health gate. Its synchronized warm medians were 38.752 seconds for
HNN and 543.734 seconds for exact gradient, a descriptive `14.031x` ratio.
This answers the narrow force-mechanics cost question. It does not replace the
missing exact retained chain and cannot support direct posterior agreement or
STR-UKF performance promotion.

## Repair Record

1. Attempt 01 was preserved unchanged after exact warm-up health failure.
2. The aggregator was repaired so performance promotion requires the matched
   mechanics health gate, not merely a lower median time.
3. Cost accounting was repaired to separate HNN preparation break-even from
   full independently tuned reuse-campaign break-even.
4. Attempt 02 reused the successful archived HNN samples, ran only the safer
   exact chain for accuracy, and reused archived HNN weights only for a fresh
   matched timing benchmark. It did not retrain or resample HNN.
5. Attempt 02 repeated the exact energy-health failure, so no further repair
   launch is justified under the phase budget and retry limit.

## Decision Table

| Field | Status |
| --- | --- |
| SIR primary accuracy criterion | passed in both arms |
| SIR direct HNN/exact agreement | passed |
| STR HNN sampler/truth criterion | passed |
| STR structural/no-noise criterion | passed |
| STR exact comparator | failed energy-health gate twice |
| STR direct HNN/exact agreement | unresolved, not computed |
| Matched mechanism criterion | passed for SIR and repaired STR mechanics |
| Second-seed rule | not triggered |
| Main uncertainty | one seed; STR exact-HMC tuning instability |
| Next justified action | terminal synthesis; no further STR retry in this campaign |
| Not concluded | universal superiority, STR direct posterior agreement, default readiness, or from-scratch total cost |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto evidence | clear for both SIR arms and STR HNN; fired for STR exact arm |
| Viable candidates | SIR HNN/exact and STR HNN |
| Statistically supported ranking | none |
| Descriptive-only differences | all speed, tuning, sampling, ESS, and break-even values |
| Default readiness | not established |
| Next evidence needed | a separately planned STR exact-HMC stabilization study if direct comparison is still needed |

## Artifacts And Budget

- SIR result: `docs/plans/artifacts/hnn-neutra-exact-gradient-comparison-repair-20260718/phase4-serious/SIR-SGQF/attempt-01/result.json`
- STR attempt 01: `docs/plans/artifacts/hnn-neutra-exact-gradient-comparison-repair-20260718/phase4-serious/STR-UKF/attempt-01/result.json`
- STR exact repair: `docs/plans/artifacts/hnn-neutra-exact-gradient-comparison-repair-20260718/phase4-serious/STR-UKF/attempt-02-exact-repair/result.json`

SIR used 1.016 GPU-hours. STR used 3.334 GPU-hours in attempt 01 and
1.260 GPU-hours in attempt 02, totaling 4.594 hours under its six-hour ceiling.
No Phase 4 cell or total campaign compute ceiling was exceeded.
