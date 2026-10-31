# D4b saved-reference assessment repair, 11 October 2026

## Question and scope

The SGQF numerical cohort finished at 01:58:39 UTC on 11 October. All six
SNAPER candidates passed tuning, posterior checks and the independent agreement
screen. All nine MAMS candidates passed tuning; five passed posterior checks.
One MAMS member exhausted warm-up without establishing readiness and produced
an empty retained tensor; three exhausted retained sampling without meeting
precision. The assessment incorrectly passed the empty tensor to
`mean_precision`, aborting assessment of valid peers. This is a reporting
failure, distinct from the four unsuccessful posterior deliveries.

Repair only the post-run diagnostic and its regression tests. Keep the original
fit, failed assessment log, source snapshots, tuning membership, thresholds,
seeds and sampling outcomes unchanged. Write reassessment and reconciliation
under a fresh `N5/method-reference-repair-01/` directory. Do not rerun sampling.

## Evidence contract and research intent

| Item | Requirement |
| --- | --- |
| Question | Can expected empty or too-short retained archives be reported without suppressing valid peers? |
| Baseline | Original nine-member MAMS result and failed assessment traceback; current missing/failed/nonfinite-member regression |
| Mechanism | Classify zero or fewer-than-four retained draws before invoking precision estimation |
| Primary pass criterion | All nine identities remain in the reassessment; empty/short members receive explicit unavailable status and later eligible peers are assessed |
| Promotion veto | Missing posterior requirements, numerical vetoes, invalid precision or failed unchanged reference screen still prevent member agreement |
| Continuation veto | Changed target/data/source/reference identity, corrupt tensor rank/chain/quantity shape or failed regression |
| Repair trigger | Diagnostic crash on a valid empty/short archive |
| Explanatory only | Elapsed time, acceptance and per-member diagnostic values |
| Not concluded | Convergence of failed members, changed tuning eligibility, method superiority or release readiness |
| Preserved evidence | Original fit and logs plus fresh tests, assessment, manifest and settled budget |

## Execution and bounded budget

1. Reproduce the fault with zero-, one- and three-draw archives followed by a
   passing peer. Include the four-draw boundary, where lugsail precision may
   remain unavailable, and malformed rank/chain/quantity cases that still stop.
2. Add the small diagnostic guard; run existing and new reference regressions.
3. With CUDA intentionally hidden, reassess all saved MAMS members using the
   original frozen reference. Verify numerical/input hashes and record original
   versus repaired diagnostic hashes. Preserve the failed original manifest.
4. Settle cost once, record outcome/decision/inference tables, and refresh the
   master and active D4 status with the repair and remaining work.

Environment: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`,
`CUDA_VISIBLE_DEVICES=-1`, `TF_FORCE_GPU_ALLOW_GROWTH=true`, two TF intra/inter
threads. Tests are CPU mechanics; reassessment is CPU post-run diagnostic
evidence, not default GPU sampling. Existing serious numerical runs are terminal.
Reserve at most 600 seconds for focused checks, 120 for the reassessment and 60
for bookkeeping: 780 seconds from the existing grant, not additional funding.
These are convenience safety ceilings, not statistical criteria; the 120-second
assessment cap is inherited from D4b. Record actual occupied durations only once.

## Default audit and skeptical pre-run review

- Four draws is the precision API's existing minimum, not a new inferential
  threshold. Its lugsail batch-validity checks remain authoritative above that
  boundary. Empty tensors signify absent retained data, never successful checks.
- Four chains and the model-specific quantity count come from the frozen cohort.
  Malformed shapes remain corruption errors; no broad exception handler hides them.
- The existing .10 reference screen, multiplier 3 and reference uncertainty are
  inherited unchanged. No posterior quantity enters tuning or orders candidates.
- Test arrays only exercise reporting; the real saved cohort tests the failure
  route. Successful process exit cannot substitute for inspecting every member.
- Strongest misleading success: reporting all nine records as agreement despite
  empty data or failed posterior checks. Assertions and the terminal audit must
  reject that interpretation. No statistical method comparison is performed.

**Pre-run audit: proceed.** The failure is a missing expected-data boundary in
the diagnostic, not sampler invalidity. The repair preserves the target, full
denominator, failed evidence, posterior criteria and remaining campaign budget.

## Execution result

The original queue finished at 01:58:39 UTC on 11 October with a partial
manifest. Its SNAPER cell completed 6/6 tuning verification, posterior delivery
and separate agreement. Its MAMS cell completed 9/9 tuning verification, but
only 5/9 posterior deliveries. The original MAMS assessment crashed on the
zero-draw member; its failed log and partial queue manifest remain unchanged.

The four new short-archive regression cases failed before the fix as expected
(4.442 seconds). After the guard, all eight focused reference tests passed
(4.485 seconds), then all 22 consumer tests passed (127.34 seconds reported by
pytest; 128 seconds from log creation to final write). Three malformed-shape
cases still fail closed. The repaired CPU reassessment finished in 3.348 seconds
without numerical sampling. It retained all nine candidate identities and
classifies one as `no_retained_draws`, three as
`agreement_not_established`, and five as `agreement_screen_passed`.
The three unestablished candidates had 10,000 retained draws per chain but
missed the existing precision target; one had no retained draws because warm-up
readiness remained inconclusive. One precision-limited candidate had an
observed reference screen below .10, but cannot be counted as agreement because
its posterior precision check failed. No candidate has a recorded numerical
hard veto. Tuning status remains verified for all nine.

The independent assessment checked the frozen source/data/reference hashes.
The original numerical consumer and proposal script hashes still match the
queue's launch manifest; only the separate diagnostic changed. The original
failed log, fit results and launch manifest remain at `N5/method-full-fit-01/`.
The repaired report, a new 20-cell inventory snapshot, ordinary SHA-256
provenance and a conservative charge ledger are under
`N5/method-reference-repair-01/`. The ledger charges 143 seconds for the failed
regression, focused and full regressions, and CPU reassessment. Combined with
the terminal queue ledger, 273066.235 seconds of the 432000-second campaign are
charged; 158333.765 seconds (43.98 hours) remain after the existing 600-second
overhead reserve. Neither the old parent duration nor its children are added
again. This is a conservative accounting charge because the four component
times were rounded upward.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close local reporting repair | 9/9 original identities present, all eligible peers assessed, 22/22 consumer tests pass | Corrupt shapes still reject; no source/reference mismatch | Single saved target cohort | Retain corrected assessment and original failed log | Posterior repair or changed tuning membership |
| Record SNAPER scoped delivery | 6/6 verified, 6/6 posterior and agreement screens pass | No recorded member veto | One seed and no fair uncertainty-supported method comparison | Carry as a viable arm into paired replication | Superiority, exact nonlinear truth or release |
| Record MAMS partial delivery | 9/9 verified; 5/9 posterior and agreement screens pass | Four delivery failures, no recorded numerical hard veto | Whether stronger target-specific proposals or a separately justified sampling allowance would repair them | Inspect per-member mechanism and budget before next cohort | Failure of the MAMS research direction |

| Inference status | Conclusion |
| --- | --- |
| Hard veto screen | No candidate numerical hard veto; four MAMS members fail declared posterior delivery; two of these also exceed the unchanged reference screen |
| Statistically supported ranking | None: one frozen seed and unequal proposal/search costs |
| Descriptive-only differences | Per-member retained count, precision, agreement ratio and elapsed time |
| Default-readiness | Not established for either method |
| Next evidence needed | Stronger classical baseline, remaining target cohorts and independent paired-seed replication under a refreshed budgeted plan |

Post-run red team: a passing reference screen on a short or imprecise chain
could falsely suggest valid posterior delivery. The reassessment requires the
posterior pass flag before calling agreement established; a zero-draw member
never enters precision estimation. Conversely, the four incomplete members do
not invalidate the five delivered peers or their verified tuning artifacts.
