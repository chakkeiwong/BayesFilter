# Containment and consensus execution checkpoint

The CPU process-containment experiment passes in03290. Fresh numerical workers
return to approximately567MiB initial RSS despite each reaching about4447MiB
after four compiled signatures. Parent RSS grows3.578MiB and its mapping count
stays fixed. Every complete original numerical comparison and process-cleanup
check passes. The [containment result](filter_gradient_process_containment_result_20260923.md)
preserves the03289 diagnostic-loader failure and explains the limits. GPU
containment remains pending; the reviewed unit has used2/4 workers and
282.823288/2400charged seconds.

The public precision-consensus endpoint now executes finite/symmetry checks,
batch symmetrization, mean/shrinkage and its final SPD decision inside a single
stable-input XLA function. Host input conversion, shape/config validation and
reporting completed rejection flags remain outside. It preserves the original
candidate-SPD criterion: an indefinite individual matrix may still contribute
to a positive consensus. No input regularization, damping or clipping was added.

One explicit validity repair rejects a nonfinite candidate caused by overflow
of finite inputs. The original eigenvalue comparison could accept NaN. The
dedicated regression reproduces that invalid original result and verifies
rejection; healthy arithmetic is unchanged.

All19 CPU endpoint checks pass in03291 and again03293 after a diagnostic HLO
variable/import lint correction. Tests compare independently extracted
3582b4ac functions at D3/two inputs and D5/four inputs; verify changed inputs,
weights0/1, exact rejection messages for13 invalid cases, derivatives through
an enclosing XLA function, one trace and identical HLO for changed operands.
The two existing public consensus/geometry tests also pass. Both policy renewals
03292/03294 pass129 checks. The endpoint unit has used4/8 workers and
34.084449/1200charged seconds. GPU preflights declined before launch; no GPU
endpoint qualification or matched-cost result is inferred.

The guard still covers228 sources and1333 exact exceptions. Its function scope
now includes the consensus wrapper, validation kernel and tensor kernel, with
no new exception. Two old watchdog exception explanations now say precisely
that the callbacks are non-JIT wall-time boundaries; they do not exempt the
surrounding numerical endpoint. Focused Ruff, critical-error runtime/driver
Ruff and whitespace checks pass.

Static inventory03295 covers3045 working-tree Python files (3044 parsed; one
unchanged historical vendor syntax failure). The
[endpoint follow-up](filter_gradient_endpoint_followup_20260923.md) records
concrete repair/acceptance paths for host exception/fallback and staged locator
boundaries. The14 partial scopes are not blanket exemptions or violation counts.

Cumulative charges through03295 are65208.20247875438CPU and
63571.041305521314GPU seconds, leaving13.886610CPU/34.341377GPU process-hours
under unchanged32/52-hour caps. The updated master and ledger retain all open
gates. The initializer clipping-report proposal still awaits agreement; no
correction or revised comparator is installed. GPU locator/containment/consensus,
clean GPU costs, public initializers, actual DZ5 integration, final endpoint
dispositions and remote integration/retest remain open. Remote main remains
47ae8836c after fetch; main is not merged.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep CPU consensus repair | Original healthy values/errors/derivatives and stable compiled reuse pass | Invalid-overflow result is rejected; no healthy mismatch | GPU path and complete final qualification | Run the registered GPU group when admitted by the selector | GPU/default readiness or cost improvement |
| Keep containment evidence | Two full original child comparisons and cleanup pass | No parent-growth trigger | Bounded CPU lifecycle only | GPU test and actual consumer integration | Native in-process eviction |
| Continue E2/E5/E6 | Concrete paths cover remaining confirmed gaps | Pending owner criterion and incomplete GPU/consumer/terminal gates | Some endpoint boundaries still require implementation | Preserve gates and proceed with independent work | Full repair completion or permission to merge main |

Post-run review: source syntax and internal kernels cannot prove public
execution; explicit host boundaries remain visible. The new consensus tests
preserve an independent original authority and distinguish numerical validation
from algorithm changes. GPU resource contention invalidates no CPU observation
but supplies no GPU evidence. Canonical LEDH rebuilding remains excluded.
