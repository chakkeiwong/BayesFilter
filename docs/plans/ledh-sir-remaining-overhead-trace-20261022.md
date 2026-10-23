# Remaining SIR merge overhead: source and compiler trace

Owner request: trace the code to understand the remaining slowdown. Work on
`ledh-refactor-with-policy-fix`, HEAD `90a3acebd`, with the retained candidate02
runtime and unrelated unfinished work preserved.

Question: which added dependencies, repeated computations and execution-lane
differences remain between pre-merge and the repaired SIR consumer? Trace the
actual consumer through flow, analytical derivatives, checked factorizations,
UKF, densities, reset and correction. Identify the earliest causal dependency,
rather than treating every added check or HLO instruction as a separate cause.

Comparators: frozen pre-merge and candidate02 sources and optimized HLO from
`artifacts/ledh-sir-merge-repair-20261022/attempt-08-premerge-timing/`.
Attempt09's archived determinant ablation supplies existing causal evidence;
attempt10 supplies the previously tested Cholesky candidate. No older invalidated
LEDH results, unrelated batch implementation, or changed controls are baselines.

Pass criterion: each implementation finding has a source call chain and, where
compiled reachability is claimed, an executable check on the actual optimized
consumer. Distinguish redundant algebra, newly live work, changed validation,
and unmeasured scheduling effects. Static counts establish reachability only;
they cannot assign runtime percentages. A missing source/HLO association or
disagreement with the preserved graph blocks that particular conclusion.

Budget: this continuation is CPU-only source/HLO analysis, bounded to 15 minutes
of diagnostic process time, no GPU allocation and no runtime candidate change.
Inputs are existing artifacts; CPU-only scripts hide GPU devices and use only
the standard library. Source hashes and analysis commands are recorded. No
default, guard, tolerance, formula or training setting changes. A new timing
experiment requires its own concrete comparison protocol within the existing
remaining allocation; this plan does not authorize repeated promotion attempts.

Skeptical review: PASS for explanatory tracing. The wrong-baseline risk is
controlled by matching frozen package files and actual consumer HLO. The main
assumption is that preserved HLO describes the measured scope; verify its run
manifest and source identity. Higher instruction counts, poor conditioning and
successful parity are explanatory, not speed or scientific promotion criteria.
No learned method is being ranked; the heuristic-dominance veto remains unchanged.
An inherited finite-check implementation is a hypothesis to inspect, not proof
that its entire cost is necessary. Keep mathematical validity separate from the
cost of the chosen implementation.

Outputs: `artifacts/ledh-sir-remaining-overhead-trace-20261022/`, with structured
source/HLO checks and a concise result distinguishing confirmed causes from
remaining uncertainty. Update the active checkpoint after substantive findings.

Completed: source and tuple-aware HLO dependency assertions pass. Both state
recurrences depend on the determinant pivot loop; tangent recurrences additionally
depend on its triangular solve. Seven compiled innovation-validation fusions
remain even after determinant ablation. Auxiliary states/tangents already existed
in the pre-merge compiled consumer. Frozen source and scope checks pass. See
[result](artifacts/ledh-sir-remaining-overhead-trace-20261022/result.md).
No runtime edit, new GPU experiment or promotion follows from static counts.
