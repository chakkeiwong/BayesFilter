# Extreme-scale precision comparison decision

Status: approved by the owner on 2026-09-21 ("I approve. Continue the execution").
The comparator is installed in the two declared boundary fixtures;
mutation/independent-reference qualification is running after the frozen costs. Historical strict failures remain preserved. This approval
is separate from the factor-output and deficient-rank condition changes.

The new D3/D5 stress fixtures fit diagonal precision matrices of scale `3e150`.
Their mathematically zero off-diagonal entries acquire roughly `1e133--1e134`
roundoff in both the original Eigen solver and current TensorFlow solver.
Comparing each such entry with `atol=rtol=1e-10` effectively requires those
roundoff remainders to agree to ten significant digits. Runs02173--02176
preserve the failed comparisons. No field has been omitted.

Independent100- and160-digit Decimal least-squares calculations in02177 CPU,
02178/02182 GPU give exactly the same binary64 reference on these designs.
Design condition numbers are1.61/1.81. Original/current precision errors are
at most4.85e-16 of the reference matrix scale; relative response residuals are
below3.90e-16. Six one-ULP perturbations of the original input design make the
original solver itself fail up to2 D3 and18 D5 entries under the strict gate.
Every non-precision field in the repaired CPU/GPU records still passes1e-10.
This separates a comparison-scale issue from the near-identity eigenvalue bug,
which was repaired and passes the original unmodified gate.

Proposed change, restricted to the two declared `huge` D3/D5 stress fixtures:
compare every `raw_precision` entry by the single matrix bound

`max(abs(candidate - original)) / max(abs(original)) <= 1e-12`.

Keep exact shapes, field sets, ranks, booleans and all decisions. Every other
field and every other dense fixture keeps the existing1e-10 comparison.
Require both original and current precision to agree with the independent
100/160-digit reference to1e-12 of its matrix scale and their response residuals
to remain below1e-12. Retain all old strict failures. Add mutation checks for
an error above the bound, missing/changed shapes and unexpected fields before
installing the comparison. No runtime code, optimizer, threshold, seed,
conditioning check or scientific decision rule changes under this proposal.

This is an explicit change in the comparison criterion. The1e-12 bound is a
diagnostic accuracy screen (over2,000 times the observed error), not a theorem
or a general accuracy guarantee. It can conceal changes below that matrix
scale, so it is deliberately limited to these well-conditioned artificial
stress fixtures with an independent high-precision reference. Default-scale
filter values, analytical gradients and all other records are unaffected.

Primary-agent review: the comparison remains complete; the independent solution
uses exact input floats and its precision is checked twice. The weakest part
is the deliberately tiny, well-conditioned design, which supports no general
claim for huge or ill-conditioned models. All numerical/cost/public consumer
gates remain required. No independent review is claimed.

| Decision | Primary criterion | Veto | Uncertainty | Next step | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Execute the approved bounded comparison change | High-precision and original self-sensitivity evidence | Strict huge raw-precision gate still failed | No general extreme-scale accuracy guarantee | Implement scoped comparator and qualify it after the frozen cost matrix | No merge or complete repair readiness |

The owner approval authorizes this explicit comparison-criterion change. It
does not authorize broader tolerances or bypass the remaining numerical and
performance checks.

Qualification attempt02231 passed21 mutation cases and failed the D5 positive
case because its newly generated design had condition2.11, outside the declared
well-conditioned fixture scope. This was a test-fixture error. The mutation
fixture now replays the exact original design stream (including the preceding
frame draw); no comparator or gate changed. Preserve02231 and rerun the full
bounded suite before accepting the implementation.
