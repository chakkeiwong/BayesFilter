# HMC consistency repair: execution audit

The [repair plan](bayesfilter-hmc-consistency-gap-repair-plan-2026-09-15.md) was
reviewed skeptically before implementation, refined during the terminal audit,
and executed against baseline `8275b497`. The F1–F10 implementation repairs are
complete. **322 final regression tests and two complete automatic-search tests
passed.** The guide builds and its changed pages were inspected. A separate
CPU/XLA smoke passed. Trusted GPU compatibility remains untested in this repair:
the readiness probe found no idle permitted GPU on both checks.

This verdict concerns the demonstrated engineering failures. It does not certify
the absence of other bugs, convergence of a posterior, target-scale runtime, or
a universal probability of tuning success.

## Repair and evidence ledger

| Finding | Resulting behavior | Discriminating evidence |
| --- | --- | --- |
| F1: lost opposing epsilon evidence | Repairs prioritize unvisited geometric interiors supported by nearby opposing observations. Finite refinement also investigates unresolved reversals when no original family survives. Every proposal still needs measurement and verification. | The deterministic `1 → 2 → sqrt(2)` history now reaches `2**0.75` under the unchanged three-repair cap. Restart, nonmonotone reversals and no-survivor refinement are covered. Both complete automatic target tests pass. |
| F2: mixed rejected/verified results unreadable | A typed decision separates acceptance compatibility, evidence validity, promotion vetoes and repair eligibility. An in-band rejected receipt is valid persisted evidence; it cannot grant verified membership. | Mixed results pass writer/reader validation. Compiled position-field traces with an in-band vetoed peer survive checkpoint restart while the healthy peer receives verification. |
| F3: divergence translation differs by adapter | Available native divergence vetoes the current pair. Otherwise valid directional evidence may propose a child. Hard execution failures still disable repair. | Exact/common translations agree for finite low and in-band acceptance with divergence. Position-field divergence remains a promotion veto; its rejected receipt round-trips. |
| F4: incomplete retained seed checks | Block, sequential and continuation checks use one seed inventory. Durable work/accounting records reconstruct every attempted chunk seed after export, including failed calls without traces. | Reuse of every completed chunk seed is rejected before numerical work. A verified member exported while a refinement peer is interrupted rejects that peer's completed and failed-attempt seeds after reload. |
| F5: expensive pending work blocks affordable verification | Unfundable work remains pending while affordable reserved stages finish. Search status remains partial. | The L=3/L=25 fixture now spends its 600-unit allowance on L=3 measurement and verification while preserving L=25. Repeated resume neither charges blocked work nor changes the verified set. |
| F6: resume double-charges completed chunks | Scheduling quotes unfinished work. Numerical adapters charge each attempted native chunk before calling it; saved completed chunks are reused. Failed native calls remain conservatively charged. | Interrupted measurement plus verification uses 5,376 units including its failed 1,024-unit attempt. Interruption between chunks uses the uninterrupted 4,352-unit allowance. Two native interruptions preserve work identity and finish at 5,504 units. These are derived fixture costs, not tuning defaults. |
| F7: scope collection depends on tuple order | Member lookup searches all matching scope/search results; optional search identity disambiguates lookup. Completeness requires every included search and every expected scope. | Reversing two searches of the same scope preserves member lookup and the incomplete collection verdict. |
| F8: late option checks and opaque preparation failure | Types, supplied L grids, evidence allocations, conflicting policies and unsupported overrides are checked before preparation. A progress file preserves preparation phases, elapsed time and failure. | Sentinels prove invalid options never enter preparation. Injected preparation failure/deadline records survive and require a fresh retry directory. Numerical restart remains available only after a frozen scope exists. |
| F9: ignored options and conflicting execution defaults | Bound adapters reject redundant overrides. Shared execution config controls stage counts. Position-field legacy adaptation count becomes an explicitly documented fixed-pair pilot count. The position-field config defaults to XLA and requires a reason for an active non-XLA exception. | Pilot counts and charged transitions match the supplied allocation. Full warmup health remains checked even when warmup is discarded from acceptance. CPU/XLA execution passes. GPU validation is still pending availability. |
| F10: inconsistent guide/registry | Registry, generated route tables, reference and guide describe the shared broad grid, measured directional repair, separate evidence roles, budgets, preparation and replay. Historical config metadata is labeled explicitly. | A test compares the actual position-field primary candidates with the registry's broad-grid description. Generated files are current. The full guide builds; PDF pages 414 and 420 were inspected. |

R-hat remains reporting-only in tuning. High, missing or failed-to-compute R-hat
does not reject, repair, rank or delay a candidate. The separate retained
posterior controller still owns its cumulative R-hat/ESS assessment.

## Structure and compatibility

The active implementation now shares small modules for decision interpretation,
epsilon proposals, and chunk seeds/deadlines/accounting. Active ordinary
preparation orchestration moved into `hmc_preparation.py`; its original public
import and signature delegate to that implementation. Historical numerical
primitives retain their existing behavior. The candidate/preparation modules
parse without NumPy imports.

Controller policy is version 3 because proposal and accounting semantics changed.
Older controller results remain readable but cannot resume under the new policy.
Source-bound numerical checkpoints likewise require their original dependency
closure. This is an explicit compatibility boundary, not a silent reinterpretation
of previously issued tuning evidence.

The broader suite revealed four tests that already failed on the committed
baseline. Three called the retired public callback route; one required a
multi-point initial grid that the shared controller no longer requires. Their
historical algorithm coverage now calls the named historical helper, and an
active-facade test verifies callback rejection. No retired tuning authority was
restored to make those tests pass.

## Validation and interpretation

All paths below are relative to
`docs/plans/artifacts/hmc-consistency-gap-repair-2026-09-15/`.

| Check | Outcome | Evidence and limits |
| --- | --- | --- |
| Final regression gate | **322 passed**, 403.79 s | `validation-r8.log` / `.xml`; controller, persistence, execution, public APIs, fixed transport, position field, guide contracts and existing NeuTra policy guards. CPU debugging with GPUs hidden. |
| Complete automatic isotropic Gaussian | **Complete; 18 verified pairs**, 86 candidates, 162 work items | `automatic-regressions-r2/test_automatic_preparation_and0/isotropic/`; 214.30 s including imports. Standard preparation/search with an explicit CPU/non-XLA exception. |
| Complete automatic anisotropic Gaussian | **Complete; 18 verified pairs**, 100 candidates, 182 work items | `automatic-regressions-r2/test_automatic_preparation_and1/anisotropic/`; 247.29 s. Exact independent Gaussian variances 1 and 9, declared scales 1 and 3. Candidate cap reached; completion is relative to that finite policy. |
| Persistent complete-search gate | **2 passed**, 471.85 s | `automatic-regressions-r2.log` / `.xml`. Includes artifact validation and all-primary-L assertions. Runtime includes both subprocesses and test overhead. |
| CPU/XLA tuning, resume, export/reload, retained block and position pilot | **Passed**, 21.94 s | `cpu-xla-r2/`; four verified exact pairs and a 72-transition position pilot including discarded warmup. Device evidence is CPU. |
| Trusted GPU readiness | **No idle permitted GPU**, twice | `gpu-readiness-r1.json`, `gpu-readiness-r2.json`. No GPU numerical launch occurred. NVIDIA inspection showed existing compute use; this is resource availability, not evidence of broken CUDA or unsupported XLA. |
| Guide build and rendering | **Passed** | `guide-build-r2.log`, `guide-render-inspection.json`, `page414.png`, `page420.png`. Three pre-existing unresolved citations remain: Gorinova2020, Pakman2014, Afshar2015. |
| Generated documentation and source hygiene | **Passed** | `generated-docs-check.log`; AST/NumPy import inspection and `git diff --check`. |
| Unrelated work preservation | **12 of 12 original hashes match** | `run-manifest.json`; unrelated SSL/pruned-SRUKF changes are outside this commit. |

The automatic fixture results support the narrow claim that the repaired search
can complete and retain passing candidates in the two tested cases. The previous
zero-member Gaussian failure is no longer reproduced. Counts, acceptance and
runtime are descriptive: source-bound candidate identities change the random
streams, and these are single-seed fixtures. The intermediate Gaussian run
retained 16 members; that difference is not evidence of improvement or regression.

| Inference status | Verdict |
| --- | --- |
| Hard veto screen | Regression counterexamples enforce health, source/identity, seed and budget invariants. Rejected parents stay rejected. |
| Statistically supported ranking | None. No paired multi-seed comparison was attempted. |
| Descriptive-only differences | Candidate counts, observed acceptance, R-hat and runtime. |
| Default-readiness | The engineering repair passes its CPU and documentation gates. GPU compatibility and target-scale numerical/scientific readiness are not established here. XLA's default follows owner policy. |
| Next evidence needed | Trusted GPU smoke when a permitted device is idle; consumer-specific target/geometry and cumulative posterior checks for research use. |

## Development failures and terminal review

The first focused regressions recorded 18 failures against the unrepaired code.
Three early L-cap cases used an invalid preparation cap and therefore did not
test the intended ordering; they were corrected to supply L=26 against the valid
25-step preparation cap. Intermediate runs exposed expected message changes,
a changed acceptance/health assertion, and legacy fixtures with an invalid
active acceptance band. Those cases were repaired without relaxing policy.

The terminal audit found an additional F4 export hole. Its regression failed
because a failed native call's seed reached the retained numerical runner.
Deriving attempted seeds from the durable accounting record repaired both live
and reloaded behavior. Warmup fault injection also verified that nonfinite
discarded draws cannot hide behind valid retained acceptance. An initial test
attempted to resume an already complete immutable result; the corrected test
pauses after the mixed measurement cohort and resumes its pending verification.

An isolated baseline test checkout initially lacked the compiled local
Sylvester extension. Copying the already installed extension into that temporary
checkout allowed the baseline comparison, which reproduced the same four
historical test failures. The first CPU/XLA harness run mixed the deliberately
non-XLA `smoke()` preset with explicit XLA execution; preflight correctly rejected
it. The second run explicitly set XLA on in both configs and passed. These are
harness/environment repairs, not candidate or method rejections.

The strongest alternative explanation for successful tiny-target tests is that
they exercise easy geometry with a narrow family of transitions. The weakest
evidence is GPU and target-scale behavior, which this run did not measure.
A failing unchanged-source restart, a reproduced invariant violation, or a
target-specific proposal hole would reopen the corresponding repair. Passing
tests are evidence for their cases, not a proof of complete implementation
consistency.

## Decision and remaining limits

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Commit the F1–F10 engineering repairs and guide | Passed: final regressions, complete automatic searches, CPU/XLA mechanics and documentation checks | No unresolved CPU correctness failure in the tested cases | Unexercised interactions and hard targets | Use the repaired interfaces and preserve any new counterexample | Universal tuning success or bug-free code |
| Leave GPU validation explicitly pending | GPU check could not launch | Readiness policy found no idle permitted device | GPU compilation/runtime compatibility of the final source | Run the preserved bounded GPU smoke when a permitted GPU is idle | CPU evidence establishes GPU compatibility |
| Keep historical extraction and large-scale persistence work bounded | Shared failure-prone logic and active preparation were extracted | No requirement to rewrite unrelated historical numerics | Large embedded evidence and recursive retained-history cost remain unmeasured | Profile representative replay workloads before an evidence-store redesign | A wholesale rewrite is required, or small tests establish scalability |

Reasonable tuning problems can still fail because geometry preparation fails,
the declared epsilon/L domain misses useful settings, evidence stays inconclusive,
numerical health rejects a pair, or finite work/candidate budgets expire. A
complete empty set means the declared search ended without a verified member;
it does not prove no suitable kernel exists. Cooperative deadlines cannot
interrupt an individual native call or compilation; hard limits remain external.

No MacroFinance-scale campaign, learned transport training, threshold relaxation,
ESS/R-hat ranking, or posterior promotion was performed. The exact commands,
source hashes, environment, timing, and artifact inventory are recorded in the
[run manifest](artifacts/hmc-consistency-gap-repair-2026-09-15/run-manifest.json).
