# Bounded legacy graph localization

Question: are the empty outputs related to loop-carried TensorArrays that are
reserved but never written when optional diagnostics are disabled?
Comparator: the exact legacy callable body in an explicitly eager CPU
diagnostic. It is not an eligible runtime fallback. Use the same tiny
d=1,N=4,T=2,FP64 inputs as the completed endpoint audit.
Run default, trace-only, Fisher-only and all-diagnostics-enabled variants;
compare scalar, six-parameter score and particle history. Also test a minimal
while-loop with an unused versus written TensorArray. Each numerical body is
the actual shared function or a explicitly labeled isolated mechanism probe.

Criterion: output shapes must match and all numerical discrepancies be <=1e-10.
A failing graph vetoes that legacy graph lane, not the new checked paper-value
path. Full returned arrays and case flags must be preserved; errors are
diagnostic rather than a replication verdict. No method ranking, paper-scale
validity, XLA conclusion or default readiness follows.

Budget: one 60-second CPU phase within the existing 1800-second total and
16-attempt tranche; deliberate CUDA_VISIBLE_DEVICES=-1 before imports.
No GPU, package mutation, pfor or NumPy. Source snapshot and exact command are
issued by the master. Observed remediation must be checked before any patch.

Skeptical self-review: PASS for localization. Matched inputs and same callable
body distinguish a lane failure from different filtering equations. Optional
diagnostics change loop bookkeeping but do not change the first three intended
outputs. Their expected equality is tested, not assumed. The minimal probe
may fail to reproduce; absence would leave the deeper root cause open.

Attempt14 rejected the unwritten-array hypothesis: all four legacy graph
variants return empty outputs, whereas both minimal TensorArray examples
return the expected scalar. No runtime patch is justified by that hypothesis.
Next discriminating diagnostic (<=60 additional CPU seconds, same total budget
and tranche): fresh-cache graphs with the meta-optimizer disabled, then each
of arithmetic, dependency, loop, function, constant-folding, shape and remapping
optimizations disabled. Restore the original options after each case. Compare
the actual kernel with the same callable body and unchanged tolerance.
These are CPU debug configurations only, not eligible defaults or XLA evidence.
Self-review PASS: one-option controls can nominate the responsible optimizer
but cannot identify its faulty transformation without further localization.

Attempt15: disabling the meta-optimizer repairs the tiny legacy graph, and the
first arithmetic-disabled check also passes. Later single-pass labels are not
independent interventions: inspected TensorFlow context.py:2213-2219 updates
rather than replaces the option dictionary, so previously disabled passes can
persist. Do not attribute those later successes to their named pass.
A final isolated check runs default, no-meta and each named pass in fresh CPU
processes, records effective options, and retains the same numerical comparator.
Ceiling 90 CPU seconds total (each child <=12 seconds); within the existing
1800-second CPU budget and attempt16. This is the same diagnostic question,
not additional scientific comparison. Self-review PASS after removing the
state-contamination confound. No global or production optimizer policy changes.
