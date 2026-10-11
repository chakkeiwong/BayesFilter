# Generic trajectory adaptation execution

Plan: [October 11 repair](bayesfilter-generic-trajectory-adaptation-repair-2026-10-11.md).
Planning commit: `26ecd4932`. This record separates engineering implementation
from the later priced statistical confirmation.

## Skeptical implementation audit

The nine-point historical MAMS grid is the development comparator, not an
adequately adapted baseline. Pilot correlations and displacement nominate
policies; they do not establish qualification or posterior delivery. R-hat is
not a tuning gate. NUTS remains excluded. The current uniform integer count
law and fixed supplied geometry remain unchanged.

The incremental implementation review found further material defects to fix:

1. Repairing a repair child from its rounded realized duration can accumulate
   drift. Preserve the first family request through every descendant and replay.
2. A lag-truncated correlation sum must not appear as an available IACT. Return
   unavailable estimates, with truncation explicitly reported.
3. A weak multiplicative update proportional to acceptance error can barely
   explore epsilon near acceptance one. Use explicit bounded geometric
   bracketing; this remains a heuristic when counts change.
4. A stopped pilot must preserve completed windows and charge attempted work.
   Recovery must bind source/configuration/backend/seeds and cannot fabricate
   missing windows or reclassify an incomplete pilot as completed.
5. A large-epsilon numerical proposal failure is not by itself evidence of a
   shared target defect. Back off within the predeclared pilot allocation;
   shared invariant/identity failures still stop affected work.

No new numerical default is promoted. Window sizes, lag support, factor
neighborhoods and budget caps in unit tests are small mechanics fixtures;
the 0.3 source coefficient and 0.9 adaptation target remain source hypotheses.
State-space confirmation needs measured device pricing and budget settlement.

## Execution contract

Run focused CPU-only mechanics/reference regressions with
`CUDA_VISIBLE_DEVICES=-1` before framework import in the `tfgpu` environment.
Compilation mode is explicit per test. These checks establish deterministic
arithmetic, preserved candidate membership, source/stream separation and actual
public-pipeline wiring, not posterior accuracy or GPU readiness. Retain the
commands and pytest reports in a fresh `N5/generic-adaptation/` directory.
Any failing invariant or source mismatch stops the dependent test/experiment
until repaired. Unit regressions are authorized routine work; no long GPU
campaign begins on the historical budget snapshot alone.

The shared checkout includes substantial pre-existing changes. Commit only
identified task changes, preserve other work, and describe dependency limitations
honestly. A plan-only commit is not a coherent runnable source baseline.

## Reviewed implementation and checks

The final isolated staged-source suite passed **99 tests in 219.07 seconds**.
It covers the duration compiler, discarded MAMS proposer, repair ancestry and
replay, actual ordinary/fixed-transport and LGSSM/nonlinear accumulation
consumers, interruption reporting, modern transition mathematics/contracts,
and the existing ChEES/SNAPER proposer. The source tree tested was
`a50bc70091614a05bc9099cc83a5965f404a6a94`; documentation execution notes are
updated after that check. The prior isolated suite passed 56 tests. A new
posterior-budget test initially omitted required seed fields (two test-fixture
failures); supplying the explicit seeds repaired it and both variants passed.
No sampler conclusion is drawn from that fixture error.

The review repaired complete target/member denominators on interruption,
typed expected pilot failures, cooperative checks at pilot and posterior
chunk boundaries, and a hard parent-process ceiling. It also removed
machine-dependent remaining time from the pilot seed's configuration hash.
Stream separation now checks actual executed qualification chunk seeds,
rather than an empty optional top-level field.

An actual isolated CPU smoke ran the validation entry point for Gaussian seed
41 with a 240-second allocation. It completed in 47.822 worker seconds,
retained all three verified candidates and assessed each through the public
posterior API. All three passed the deliberately short diagnostic policy.
This establishes wiring, not posterior correctness or robustness. GPU devices
were intentionally hidden; this is not GPU/default execution evidence.

The isolated source needs the existing compiled TensorFlow Sylvester operator;
its SHA-256 is
`2feb93135f2359e60aca446f52f8468b104199b4d398d4fe6057030df20f23d6`.
It was copied explicitly as an environment dependency, never loaded from an
unstaged Python source fallback. The standalone modern chapter (40 pages) and
tuning chapter (45 pages) compiled; the new sections were rendered and
inspected. The tuning-only wrapper naturally leaves two cross-chapter
references unresolved; the full monograph build is checked separately.
The full monograph then compiled successfully at 625 pages after its existing
tracked figure/appendix dependencies were included in the isolated snapshot.

Records are under `N5/generic-adaptation/checks-01/`. Exact focused command:

```bash
env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_hmc_trajectory_duration.py tests/test_hmc_mams_trajectory_proposals.py \
  tests/test_hmc_duration_preserving_repair.py tests/test_mams_adaptation_consumer.py \
  tests/test_mams_adaptation_validation.py tests/test_modern_hmc_kernels.py \
  tests/test_modern_hmc_contracts.py tests/test_hmc_trajectory_proposals.py
```

## Bounded execution decision

The [bounded design](bayesfilter-mams-adaptation-bounded-validation-2026-10-11.md)
survives the skeptical audit after those lifecycle repairs. Allocate no more
than 7200 aggregate worker-seconds, including retries, to eight-target
feasibility/pricing. Existing regression work is covered by the conservative
3600-second reserve. The earlier 158333.7646589242-second balance therefore
leaves 147533.7646589242 seconds unallocated pending settlement. This is
existing authorized compute, not a new grant.

GPU 1 passed the trusted probe at 15:39:15 UTC on October 11, with memory
growth verified and GPU 1 idle. Use the scoped commit's source snapshot and
the existing operator dependency. The exact next command, from that snapshot,
is:

```bash
env CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/validate_mams_adaptation.py \
  --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/gpu-feasibility-01 \
  --targets gaussian rotated_gaussian student_t funnel_noncentered residual_funnel mixture LGSSM NL-ACC \
  --seed 411 --max-wall-seconds 7200
```

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Execute bounded GPU feasibility | Engineering regressions and CPU public wiring pass | No unresolved focused invariant failure | Cross-geometry behavior and actual filter cost | Run eight explicit targets and preserve every outcome | Robustness, superiority, release readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Focused invariants pass; future GPU failures remain explicit |
| Statistically supported ranking | None |
| Descriptive-only differences | Short Gaussian diagnostic delivery and observed cost |
| Default-readiness | Experimental optional path; not promoted |
| Next evidence needed | Analytic corners, remaining mutation checks, priced state-space paired confirmation |

Strongest alternative explanation: the Gaussian wiring smoke is easy and
cannot expose residual geometry or missed modes. The partially whitened
funnel and local-start mixture test that limitation. Full model observables,
independent filter references and paired replications remain necessary for
D4R4; bounded feasibility cannot close them.
