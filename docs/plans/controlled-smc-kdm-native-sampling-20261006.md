# KDM native sampling continuation — 6 October 2026

Authority: the accepted replication master and existing resource grant.
Question: does the TF GPU/XLA sampler implement the released bearings KDE
mixture draw, including circular draws and stopped sampling gradients?

Comparator: pinned Younis kernel_density_estimator.py:14-53,146-188 and
von_mises_full_dist.py:8-11; its sample method inherits the installed PyTorch
VonMises.sample and _rejection_sample. Preserve raw categorical probabilities,
normal scales, concentration=1/(bandwidth+1e-8), internal double precision,
angle wrapping and overwrite-on-accept behavior. Record the installed Torch
sampler file hash. The source samples every component before gathering;
preserve this structure in the first port.

Primary engineering screen: actual author KDE.sample replay with captured
ancestors, normal component draws and rejection uniforms; match samples,
acceptance completion and loop count, for seven concentrations and both
float32/float64. Bounds remain 1e-9*(1+abs(ref)) and
128*2^-23*(1+abs(ref)). Verify native GPU placement, one trace, stopped
sampling gradients, probability-zero components, finite outputs and no
unacknowledged rejection cap. Invalid inputs and an exhausted cap must be
flagged; the caller must not accept such samples.

Native stochastic diagnostic: 32768 samples in each of seven fixed regimes,
kappa=1e-6,1e-4,0.2,3.74,3.76,50,1000; locations include the angular seam.
Compare first and second normal mixture moments, sine/cosine moments and
component frequencies with analytic expectations, separately for source/TF.
A six-standard-error screen is a predeclared diagnostic, not a distribution
identity proof or the final no-worse acceptance rule. Use exact normal-mixture
moment variances and the conservative bound 1 for circular-feature variance.
Zero-probability categories must never occur. Report deviations without ranking.

Default/assumption audit: the concentration grid is a mechanism-based stress
fixture spanning the source Taylor branch, polynomial branch, ordinary and
concentrated tracking regimes, not a training default. Three mixture components
and 25 replay samples make the smallest auditable fixture. Native sample count
is a bounded precision/cost choice, not a final claim size. Native random
streams need the same laws, not identical integer-seed realizations. A
128-iteration cap rejects only; its no-fire status and iteration count are
mandatory. It is not an altered accepted result or a silently clipped draw.
The port adds no source repair, projection, ridge or density modification.

Budget: at most four attempts, 1800 CPU wall seconds and 1800 GPU wall seconds,
charged once to the existing balances. Fresh directories
artifacts/controlled-smc-kdm-replication-20260928/kdm-sampling-20261006-NN.
CPU source sets CUDA_VISIBLE_DEVICES=-1. GPU uses trusted permissions, verified
memory growth, TF32 off and explicit stable signatures. Preserve commands,
source hashes, environment, seeds 2026100611–2026100614, outputs and wall time.
Stop for an invalid reference, changed method, corrupt evidence or budget.
A localized infrastructure or mathematical mismatch triggers investigation
within this cap; it does not reject KDM as a research direction.

Skeptical audit PASS: actual source sampling is inspected, including the
non-obvious double precision and accepted-sample overwrites. Replay alone
cannot check native RNG laws, so native analytic screens supplement it.
Moment screens cannot establish equality of distributions; the source
operation audit is essential. This phase does not complete the neural
sampling call chain, full training/stopping/checkpoints, or stochastic
qualification. Native training wiring and batch-size-64 cost are next.

Attempt 01 stopped before the source computation because the generic interpreter
did not contain the pinned source environment's tqdm. This is an environment
failure, not sampling evidence. The phase launcher now uses the verified
mdpf-reference-20260924 interpreter and explicitly hides CUDA for the source
reference, as in the completed neural phase. No sampling code or acceptance
criterion changed.

Attempt 02 completed: 28/28 replay, native-moment, validity, placement and trace checks pass in both dtypes. The source CPU reference used the pinned interpreter. XLA emitted ptxas register-spill warnings during compilation; they did not produce invalid outputs. The phase remains diagnostic and replication incomplete.
