# Next model scope: LG2 reference preparation

The continuing research question is whether an imperfect full DCTR proposal
field supports reliable endpoint-corrected parameter inference. LG2 is the next
linear scope in the authorized all-model queue: two states, two observations
per time, T20, and two unknown parameters. This preparation supplies the exact
endpoint/reference building block; it does not admit a DCTR consumer or sampler.

Use the existing `linear_fixture_spec("LG2")`: F=a[[.7,.2],[.1,.6]],
Q=exp(2b)[[1,.3],[.3,.8]], H=I, R=.5I, stationary mean-zero initial law,
a uniform on(.5,1.2), b uniform on(-1,1), and transition before observation.
The fixture truth is(1,0). These choices are existing model definitions, not
transferred LG1 tuning. Stationarity holds over the support because the largest
absolute eigenvalue of the base transition is.8 and1.2*.8=.96<1.

For a batch of parameters, solve the stationary discrete Lyapunov equation
P=FPF'+Q through vectorization, then use the full matrix Kalman recursion.
The independent check constructs the joint observation Gaussian covariance:
Cov(y_t,y_s)=H F^(t-s) P H' for t>=s, transposed otherwise, plus R when t=s.
Check likelihood equality and total parameter derivatives, including the
stationary initial law and chart Jacobian, against that independent formula.
Check single-row/batch parity and permuted-row independence. TensorFlow/TFP is
the numerical backend; stable-signature XLA is the default. No pfor or NumPy.

Current stage is a small CPU-only reference/debug exception, with CUDA hidden
before import, at most120 CPU-worker seconds. It may run alongside C GPU tuning
and is charged to the authorized campaign. No GPU or inference-quality claim
comes from CPU timing. No new dataset, posterior tuning or claim chain is
authorized by an implementation test alone. The subsequent preparation step
must freeze data partitions and posterior function definitions before HMC.

Evidence contract: compare same-law Kalman log density/gradient with the joint
Gaussian oracle at nominal and boundary-near parameter probes. Likelihood
difference <=1e-9 and normalized gradient difference <=1e-8 pass this numerical
check. Nonfinite results, stationarity failure, wrong prior/chart/timing or
batch coupling veto this implementation. Timing is explanatory. Passing does
not establish DCTR approximation accuracy, surrogate-HMC convergence, heuristic
dominance or all-model coverage. Preserve the focused test log, timing, source
hashes and result under a new versioned campaign artifact root.

Default audit: stationary initialization and bounds come from the actual
fixture and must match the later DCTR consumer; changing either creates a new
target. FP64 is a reference exception justified by a same-law numerical oracle.
The dense covariance oracle is restricted to tiny probes to bound memory; it
cannot be substituted for the scalable runtime. Probe tolerances are numerical
checks with an independent formula, not posterior promotion thresholds.

Skeptical audit: LG1 posterior agreement would not certify this two-parameter
model; the parameter-dependent stationary covariance and process scale create
new derivative terms. Checking a constant covariance or only the transition
matrix would test the wrong endpoint. The joint-law comparison includes both.
The next serious run still needs scope-specific full-DCTR tuning, public HMC
verification, independent posterior integration, cost forecasting and the
master program's three cheap adversaries. No result here relaxes those duties.
