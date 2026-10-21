# M10 quadrature repair for the surrogate-HMC reference

The tail screen invalidated the uniform state grid as a full-prior reference.
Test a deterministic independent quadrature of the same two-observation
likelihood. The full DCTR force and its controls are unchanged. A successful
repair supplies a reference candidate; posterior normalization, parameter-tail
bounds and retained surrogate-HMC comparisons remain separate obligations.

Let f(x)=a*x+c*sin(x), h(x)=H*x+b*x^2, and p1 the density after the first
transition from x0~Normal(m,P). The actual likelihood is
integral p1(x1)*Normal(y1;h(x1),R)*integral Normal(x2;f(x1),Q)*
Normal(y2;h(x2),R) dx2 dx1. There is no filtering approximation here;
finite quadrature approximates this two-dimensional integral.

Because a=.62 and c=.2, f'(x)>=.42>0. If U=f(x0), its density is
pU(u)=Normal(f_inverse(u);m,P)/f'(f_inverse(u)). Therefore
p1(x)=E[pU(x-sigma_Q*epsilon)] over standard-normal epsilon. When process
noise is small, approximate this smooth convolution with normal quadrature;
when process noise is larger than(a+abs(c))*sqrt(P), use the equivalent
p1(x)=E[Normal(x;f(x0),Q)] with initial-normal quadrature. Both compute the
same exact convolution before quadrature. Evaluate convergence on both sides
of the deterministic switch. Inversion uses16 Newton iterations, initial u/a;
retain its residual and reject residual exceeding1e-11*(1+abs(u)). The bound
on f' prevents a vanishing inverse derivative. No clipping changes the law.

Use composite Gauss-Legendre quadrature, with intervals split at both real
roots of h(x)=y, the vertex, and root-neighborhood offsets at0,.5,1,2,4,8,16
observation widths. Width w solves abs(h'(root))*w+b*w^2=sigma_R; the stable
positive formula is2*sigma_R/(sqrt(h'^2+4*b*sigma_R)+abs(h')). Include the
conditional-normal center and offsets at0,1,3,6,10 standard deviations.
Clip and sort interval endpoints; duplicate endpoints have zero weight.
If y is below the vertex, use the vertex for both root centers. This changes
only the quadrature partition, not the likelihood. The outer state limits
are f(m±10*sqrt(P))±10*sigma_Q; the inner limits are f(x1)±10*sigma_Q.
Before quadrature error, omitted probability is bounded by6*Phi(-10), and
omitted likelihood by that quantity times the two-observation maximum
likelihood envelope. Both quadratic-observation branches must be integrated.

The refinement ladder is12/20/20/32 Legendre nodes per interval and64/64/128/128
normal nodes. These are reference hypotheses. Compare log likelihood and
total noise derivatives at the previous116 tail probes, the nine local
probes in each partition, and switch-adjacent points. Primary repair screen:
20/32 Legendre differences and64/128 normal differences at fixed20 Legendre
nodes<=1e-6 for log likelihood and1e-5 for scores; agreement with
the qualified state-grid overlap at the same thresholds. Twelve-node results
are explanatory. Finite-difference scores, inverse residual and finite
positive likelihood checks are numerical vetoes. Passing the screen does
not qualify all parameters or create an HMC tuning artifact.

Use TensorFlow FP64/XLA with stable signatures. The reference derivative is
autodiff of this explicitly named quadrature, including its moving nodes;
it is not a DCTR score and cannot replace the canonical analytical recursion.
No Jacobian APIs, pfor or training. CPU-only independent-reference exception,
GPU hidden, threads2/1. Budget200 CPU-worker seconds total including focused
tests, one initial execution and at most one localized infrastructure retry;
wall180 seconds for execution. Charge M10's existing1800-second CPU ceiling.
Use fresh `surrogate-hmc-all-models-20260921-m10-quadrature-*` roots, source
snapshots, full per-point output and failure/accounting preservation.

Skeptical audit: moving nodes can produce wrong scores if differentiated
partially; the implementation must differentiate the whole finite integral.
A convolution switch can introduce finite-quadrature jumps; evaluate both
representations around it. A single-root approximation would omit a real
branch; retain both and compare large-noise probes. All defaults above are
prospective numerical hypotheses whose early diagnostics are stated here.
No learned-method ranking or heuristic-dominance claim is made by this repair.

Pre-execution implementation audit: two focused CPU/XLA tests passed after
unrolling the fixed16 Newton iterations to avoid this TensorFlow version's
DT_VARIANT gradient failure inside a conditional. Local values, total scores
and a central finite-difference check pass. The forced-convolution diagnostic
now reports the branch actually used. The independent normal refinement above
was added before execution because fixed128-node convolution error could be
shared by both Legendre resolutions. No scientific threshold was relaxed.
