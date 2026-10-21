# M10 reference tail screen before surrogate-HMC sampling

The question is whether the current513-node, radius9 likelihood reference
remains usable outside the nine local probes. The scientific goal remains
testing surrogate HMC with the full DCTR proposal field. This screen determines
the reference repair needed for that test; it cannot promote a posterior or a
sampling configuration.

Use the same M10-T2 law, independent Normal(0,log(2)^2) log-noise priors and
frozen calibration/validation nominal/stress data as the local reference.
Keep claim data untouched. Evaluate the Cartesian product of log-noise scales
at(-3,-2,0,2,3) prior standard deviations, plus the four axial points at
plus/minus6 standard deviations and zero on the other coordinate. These
explicitly cover narrow/wide process and observation noise. Compare257,513
and1025 state nodes at radius9 and1537 nodes at radius13.5 (the last two have
the same spacing). These extend the original numerical reference ladder;
they are diagnostic hypotheses, not globally qualified defaults.

The early failure modes are narrow observation-density aliasing, unrepresented
large-process-noise state tails and inaccurate small-process transitions.
For each point retain log likelihood, analytical noise score, predictive mass
loss and posterior edge mass for every resolution. A difference exceeding
1e-6 in log likelihood or1e-5 in score for513/1025 or1025/1537 comparisons, predictive mass loss or edge mass
exceeding1e-8, or any nonfinite result vetoes use of this reference at that
point. The257/513 comparison is explanatory only. These are reference vetoes/repair triggers, not continuation vetoes for
surrogate HMC. Passing finitely many probes cannot qualify the full prior.

Also compute the elementary likelihood envelope for the two observations.
Writing s=log2 and sigma_R0=exp(-.6), each Normal observation density is at
most(2*pi)^(-1/2)/sigma_R, so
L(lambda_Q,lambda_R)<=C*exp(-2*lambda_R), C=(2*pi*sigma_R0^2)^(-1).
Multiplying by the prior and completing the square gives
C*exp(2*s^2)*Normal(lambda_Q;0,s^2)*Normal(lambda_R;-2*s^2,s^2).
Report the integral of this envelope outside the boxes[-k*s,k*s]^2 for
k=1,2,3,4,5,6,7. These are absolute upper bounds on omitted unnormalized
posterior mass. They are not posterior tail probabilities: a qualified lower
bound on the evidence is still needed to normalize them. The derivation
uses the actual two-step observation law and does not change prior support.

Skeptical audit: a local mechanics pass could conceal substantial tail errors;
using the same quadrature to generate a reference and acceptance values would
then give misleading agreement. This screen deliberately cannot certify
model-posterior correctness. Its cheap conditional probes can expose a need
for a different numerical integration method before expensive posterior grids.
The comparators are independent state-grid refinements, not learned methods;
no heuristic-dominance or stochastic ranking claim is made.

Run a CPU-only diagnostic with GPU intentionally hidden, TensorFlow FP64/XLA,
fixed signatures and two/one threads. Budget150 CPU-worker seconds and150
wall seconds across the initial run and at most one localized infrastructure
retry, within M10's7200 GPU/1800 CPU total. Preserve partial rows if stopped.
Output:`docs/plans/artifacts/surrogate-hmc-all-models-20260921-m10-tail-screen-01`.
Save exact command/environment, source/data hashes, parameters, per-point
results, envelope derivation inputs, wall/CPU accounting, and a result note.
Stop for resource exhaustion, source/data drift, malformed output or an
implementation failure; a numerical reference failure is the intended signal
for planning a bounded repair.
