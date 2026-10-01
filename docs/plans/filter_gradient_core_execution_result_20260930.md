# Core filter execution qualification

The bounded core qualification passes in its declared source and fixture
scope. Runs05139--05149 pass170 CPU reference checks. The corresponding170
GPU checks pass after two fixture-scope repairs, with one additional actual
default FP64 GPU/XLA factory comparison. CPU pool and import isolation add14
checks; terminal readback and policy run05169 passes162 checks. Numerical
runtime sources are unchanged from722b6d963. No numerical tolerance, RNG,
algorithm, runtime TF32 default or policy allowance changes in this unit.

The runner manifests, JUnit files and logs preserve commands, source hashes,
environment, seeds/fixtures, device provenance and elapsed time for all31
workers05139--05169. CPU references hide GPUs; GPU workers use the trusted
GPU3 UUID and verify growth before initialization. The verified archive is
`artifacts/filter-gradient-repair-20260917/core-execution-05139-05169-evidence.tar.gz`;
its matching verification JSON records every reopened member hash. Raw evidence
remains under the same artifact root in the main workspace.

| Scope | CPU run/checks | GPU run/checks |
|---|---|---|
| Contract E derivatives | 05139 /3 | 05150 /2 retained passes;05154 /1 repaired reference |
| Actual default Contract E factory | separate GPU witness | 05155 /1 |
| TT value/maps/analytical adjoint/actual-SV | 05140--05143 /28 | 05156--05159 /28 |
| Scalar TT | 05144 /12 | 05160 /7 retained passes;05161 /5 repaired comparisons |
| Frozen APF | 05145 /12 | 05162 /12 |
| Preparation/signatures | 05146--05147 /85 | 05163--05164 /85 |
| SGQF/streaming mask | 05148--05149 /30 | 05165--05166 /30 |
| Batched CPU pool/import isolation | 05167--05168 /14 | CPU-only contract |

Three failed workers remain failed evidence.05150 applies a CPU-derived eager
FP32 ULP limit under GPU TF32 and fails at1153/760 ULP. Frozen September17 and
current numerical closures produce identical saved primal/manual/AD fields
with TF32 enabled (05152) and disabled (05153); the latter gives0/0 ULP.
The original3/2-ULP reference now explicitly disables TF32 and restores the
prior setting.05154 passes. This does not qualify eager TF32 score accuracy.
05151 failed before numerical execution because the frozen loader lacked a
package namespace; the retry preserves its frozen numerical authority.

05155 separately executes the actual default FP64 prepared factories on
GPU/XLA at two dynamic theta inputs, with exact statuses, valid charts, a
single trace, HLO and no callbacks. Maximum absolute error is
1.0408340855860843e-16 at the unchanged1e-10 bound. The earlier eager reference
diagnosis is therefore not being used as enclosing-XLA evidence.

05160's five scalar-TT failures precede filtering: GPU preparation recomputed
a logarithm4.4408921e-16 away from the CPU frozen input. The corrected fixture
checks current CPU preparation against the original operands exactly and
feeds those same operands to the candidate on its selected device.05161
passes all five comparisons without changing value/score bounds. Cross-device
bit identity of elementary logarithms is not established or required.

The GPU matrix also stopped once between workers because terminal-test
registration changed its source fingerprint. No worker failed there. The
remaining five groups then ran with frozen sources in core_execution_finish.
Readback05169 verifies manifests, JUnit, numerical-source identity and device
policy, including the nine retained passing cases from failed workers.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action / nonclaim |
|---|---|---|---|---|
| Close this core qualification unit | Declared endpoint/reference tests and default-factory witness pass | Three failures retained and attributed; original limits preserved | Finite fixtures and inspected callbacks, not arbitrary consumers | Reuse unchanged source scopes; no whole-repository or canonical admission |
| Keep resource acceptance open | This unit contains no matched cost experiment | Existing RSS/timing triggers remain | Uncontended GPU costs/capacity and native residency | Execute the prepared bounded resource plan |
| Integrate on the repair branch | Source checkpoint is reviewable | Main merge remains withheld | Incoming SQMC changes add affected paths | Preserve parent-specific contracts and repair new execution violations |

Primary-agent result review: the strongest misleading interpretation would be
to count diagnostics as default execution, suppress partial failures, or reuse
these passes across changed numerical dependencies. The readback and separate
FP64 factory witness address the first two; manifests constrain future reuse.
Resource findings, GenUT affected-use disposition, incoming SQMC execution and
terminal F01--F20 dispositions remain separate. Adaptive iAPF/KDM and the
canonical LEDH rebuild remain deferred/excluded. No training or HMC ran.
