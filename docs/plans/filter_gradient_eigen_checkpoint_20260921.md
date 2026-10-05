# Sequential eigensystem and resource checkpoint

Through run 01839 on `repair/filter-gradient-xla-validation-20260918`, based on
`ee2d455f`, committed and pushed as `bd36a89b`. This checkpoint repairs two sequential XLA eigensystem consumers
and the resource retention exposed by that repair. The campaign is incomplete;
main is unmerged and the public outer lifecycle still uses its existing loop.

The trust-region and symmetric score-fit programs now use the existing refined
binary64 eigensystem from `mass_matrix_tf`. The eigenproblem, spectral derivative,
floor/ridge policy, optimizer settings, seeds, acceptance rules and tolerances
are unchanged. Its custom derivative is traced in an independent, fixed-signature
XLA graph under `tf.init_scope`, then called from the consumers.

Directly tracing that custom derivative in the nested `evaluate` graph retained
the graph's `bayesfilter_factor_domain_resources` dictionary and one validity
variable. Run 01823 records the ownership chain. TensorFlow 2.19.1's
`python/ops/custom_gradient.py:512` registers a closure retaining the traced
result and pullback. After graph isolation, 01825 and 01828 show that the actual
consumer graphs and all their tracked validity variables are released on CPU
and GPU, while concrete functions still execute after Python factory eviction.
This does not establish native executable-cache eviction or general leak freedom.

## Numerical authority and validation

Original source `3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf` remains the numerical
authority, with its complete dependency closure loaded independently. The
intermediate `cfbc32d2` has a demonstrated sequential eigensolver residual;
its records remain archived and cannot serve as the sole precision authority.
The original schema comparison removes only the later `jit_compile` metadata
inside factor diagnostics, with every numerical field and event still checked.

| Evidence | Result |
| --- | --- |
| 01800, before correction | Four exposed eigenconsumer failures; three companion cases pass. |
| 01801/01802 | Seven original-reference/identity cases pass on CPU and GPU. |
| 01807/01808 | Seven complete three-way terminal records pass against original source on each device. |
| 01810--01817 | All eight original full lifecycle records and target-call orders pass on each device before graph isolation. |
| 01824/01827 | All eight final eigenconsumer, spectral-pullback and changed-input checks pass on each device. |
| 01825 CPU / 01828 GPU | HLO runtime inputs, unchanged HLO, one trace, execution after factory eviction and resource release pass. GPU D3 records pass; D5 changed-input numerical records fail on both devices. |
| 01830--01834 | All 153 affected GPU preparation, score-fit, sequential, block and locator cases pass. |
| 01835 / 01839 | All 67 policy/controller cases pass. |

The D5 comparison still fails 66 fields on CPU and 53 on GPU at the unchanged
`atol=rtol=1e-10` gate. Failures concern fitted covariance, precision, loadings
and derived diagnostics. The enclosing initial-input records and progress
comparisons pass. Archived attribution against 01754 establishes that all 66
CPU failed fields predate the sequential eigenpair repair. The correction
introduces no new failed field in that comparison.

Run 01826 crosses original/current preparation with original/current fitters
and reproduces both archived own-data results. Compared with original/original,
current fitting on original data fails four fields, original fitting on current
data fails two, and current/current fails 66. Prepared inputs differ by at most
`1.1883e-16`; all four fits use 73 iterations and 217 objective evaluations.

Two bounded interventions do not close this gap. In 01829, supplying the
original initializer leaves 23/77 failed fields for original/current data
relative to the corresponding original fitter. Identical-state gradient
differences are at most `3.134e-17`. A native-loop translation of the original
two-factor decoder passes value/pullback checks, but its nested conditional
fails under ForwardAccumulator in 01836. Restricting that diagnostic injection
to the objective/final covariance, with the existing Jacobian decoder retained,
runs in 01837 but still leaves 4/53 failed fields. Neither intervention is
installed in runtime. These observations support rounding sensitivity across
the optimization; they do not prove an exhaustive causal explanation or waive
the full-record gate.

## Memory, scope and review

Prior cost matrices through 01791 are preserved as intermediate evidence.
They showed substantial host compilation overhead and a small D3 GPU allocator
peak exceeding the 2x investigation trigger. Resource ownership is now repaired
in the tested scope, but complete original-source cost measurements have not
been renewed because numerical qualification remains incomplete. No new speed
ranking or memory-acceptance claim follows from resource release.

The refreshed syntax inventory 01838 covers 2,920 working Python files, with
2,919 parsed and the one inherited external-reference parse error. Syntax
counts are search leads, not violation verdicts. The policy guard passes for
198 reviewed sources and 1,276 exact exceptions; coverage remains partial.

Primary-agent review checked the two changed call sites, unchanged eigenpair
and derivative arithmetic, helper graph ownership, complete-record assertions,
reference isolation and diagnostic job classification. The strongest remaining
alternative explanation is that coupled preparation/initializer/objective
rounding creates a broader equivalence issue than the two sampled states expose.
The complete changed-input failure is retained to test that possibility. This is
a source review, not an independent external review.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve the localized correction in a branch checkpoint | Eigenconsumer/reference and resource checks pass; affected consumers pass | D5 full-record numerical comparison fails | Full optimizer arithmetic sensitivity | Continue discriminating numerical work without retuning or tolerance changes | Complete execution repair or readiness to merge |
| Keep public outer integration gated | Complete original parity, resource and cost evidence required | Numerical and cost gates remain open | Real external progress deadlines and complete compiled cost | Finish qualification, then public reporting/wiring and independently bounded external consumers | Live callback interruptibility, HMC readiness or canonical LEDH status |

The exact commands, source hashes, seeds, environment, process time and device
provenance are in `run-01800` through `run-01839` under
`docs/plans/artifacts/filter-gradient-repair-20260917/` in the main checkout.
All runs used the existing campaign driver, sequential workers and bounded
timeouts. GPU runs used physical GPU2 (RTX 4090), TensorFlow 2.19.1, verified
memory growth and the managed-session trust designation. CPU reference workers
hid GPUs explicitly. No package, OS, external source or source-pin changes were
made. Charged totals through 01839 are 41,913.463782358 CPU / 115,200 and
37,772.450332083 GPU / 187,200 seconds; the authorized caps are unchanged.

The next work remains original-source numerical qualification, explicit
disposition of obsolete intermediate precision comparisons, renewed original
costs, public outer/reporting integration, external watchdog compatibility,
broader block/quadratic controller repairs, actual DZ5 target/transition-block
evidence and terminal F01--F20 decisions with repeated comparisons.

## Original-source authority disposition, September 21 continuation

The exact intermediate-reference groups `lifecycle_actual_*`,
`lifecycle_runtime_*` and `refinement_*` compare numerical records against
`cfbc32d2`. That source has the measured eigensolver error preserved in
01800--01808; requiring repaired precision to equal that error is the wrong
numerical authority. Preserve the tests and their artifacts as explicit
historical comparisons. They cannot replace the original-source gates or be
silently relabeled as successful numerical qualification.

Mandatory replacements retain the same complete records, cases, target-call
order, changed-input/resource assertions and `atol=rtol=1e-10`. The existing
`lifecycle_original_*` and `lifecycle_original_runtime_*` supply these checks
for outer recurrence and resource use. Add `refinement_original_*` using the
exact same refinement harness with the entire `3582b4ac` dependency closure.
Only added `jit_compile` metadata is normalized using the existing original
schema rule. No numerical field is omitted, and the D5 changed-input failure
stays a mandatory promotion veto.

The driver must bind every retired group to a named mandatory replacement,
with enforcement against missing or explanatory-only replacements. It must
continue to require the terminal original-source and three-way tests. Run all
six original refinement cases on CPU/GPU under the existing limits, then the
policy/controller checks. Preserve a failed original comparison as a blocker;
do not waive it because the corresponding intermediate comparison passed.

Skeptical review: changing an authority after observing failure risks hiding a
regression. Here the independent original authority predates all repairs, the
intermediate error is reproduced and archived, every current field remains
checked, and the known D5 numerical veto remains in force. This is an explicit
correction of reference authority, not relaxed precision or a completion claim.

01863 catches a reference-harness extraction error before numerical execution:
the original source inlines its search cloud/selection while the intermediate
excerpt starts at a later `_search_exact_candidates` call. Bind the exact
original body from `cloud_builder` through the same pre-progress boundary,
retaining its original namespace and all arithmetic. Assert the fixture's
unscaled search count and disabled movement-report option used by this step
harness; complete lifecycle tests separately retain movement reports. Retry
under the unchanged 300-second focused ceiling.

01864/01865 pass the three original symmetric cases. 01866--01868 pass the
three original factor cases on CPU, and 01869 passes all six cases on GPU2.
01870 passes all 68 policy/controller checks, including missing or waived
replacement rejection. All 16 intermediate groups now have explicit mandatory
original-source replacements; the historical tests/artifacts remain available.
This resolves reference disposition only. D5 changed-input complete-record
parity, public outer integration and original complete cost renewal remain open.
