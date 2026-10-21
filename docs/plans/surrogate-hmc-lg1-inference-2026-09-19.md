# LG1 attempt 4: reference-endpoint surrogate HMC

The question is whether full imperfect DCTR forces sample the LG1 model-parameter
posterior when acceptance uses the qualified Kalman endpoint (B/R). Exact-force
HMC (A/R) diagnoses sampler geometry. This is the first mechanism pilot; C/D,
D/R, C/R and all other models remain separate obligations. No filtering or
force-error threshold substitutes for inference.

Use the attempt4 claim dataset from the qualified master reference, the full
N1024/T10 FP32/TF32/XLA DCTR route, and the first controls selected by the separate
calibration/validation curves. Load and validate their source and scope bindings.
The FP64 integrator and Kalman endpoint are the declared reference exception.
The force has position-only semantics and original force-valid status. No field
retuning inside HMC, claim-data selection of DCTR controls, or numerical clipping.

Tune A and B independently through public `tune_hmc_kernel` with
`VerifiedTensorFlowHMCConfig`: covariance windows32,64; step adaptation128;
candidate verification64; leapfrog candidates1,2; target acceptance.7; initial
step.2; covariance shrinkage.1, jitter/floor1e-9 and condition cap1e8. These are
finite-budget hypotheses inherited from the qualified Gaussian tuner, not
model-general defaults. In 1D covariance shrinkage leaves its diagonal unchanged;
the recorded metric and eigenvalues expose a material ridge/floor effect.
The fixed metric/step/L must then pass separate 250 discarded warmup and 1000
fresh verification draws per chain. Starts z=(-2.5,-.8,.8,2.5) span prior regions;
the nominal center0 and scale1 are only adaptation warm starts. Fixed independent
seeds and all warmup/verification draws are archived. Acceptance screens use the
full [0,1] band: finite health, all-chain movement, energy error<=1000 and modern
R-hat<=1.01 decide handoff. R-hat failure is a tuning/precision repair trigger;
it is not evidence that surrogate HMC is mathematically invalid.

Cost audit: the measured full route costs1.973s for three one-step diagnostic
transitions after32.094s compilation. For planning, allow1.4s per L<=2 transition
and180s new compilation. Nomination is bounded by96+2*(128+64)=480 transitions;
fresh checking1250; the planned retained schedule1000 then2000 cumulative is
reserved before sampling. The conservative total is
`1.5*(180+1.4*(480+1250+2000)+120)=8283s`, below the3GPUh LG1 allocation.
Replace this estimate by measured scoped tuning cost before retained launch;
veto a launch that no longer fits. A/B tuning gets a2400s wall limit, and consumes
budget even if no artifact passes. Reserve2.5CPU-workerh for LG1 by moving1.5h
from the2h shared repair/analysis reserve; the total remains15h. This reflects
measured CPU work concurrent with GPU work. Keep the remaining ready-model queue.

Measured pre-retained update: scoped A and B both passed, taking 23.055s and
1092.511s respectively. Using their entire tuning time against only the 1250
fresh transitions, the retained forecast is
`1.5*(23.055+1092.511)/1250*2000 + 16*32 + 120 = 3309.4 GPU seconds`.
This includes both arms, all 2000 draws, recompilation of every 250-draw chunk,
and analysis. The CPU forecast is 1.2 times this value, 3971.2 worker seconds.
Use a 3600s wall watchdog and a separate 4800s CPU watchdog. Including the
1579.6 GPU / 1747.1 CPU seconds already charged (plus recorded CPU regressions),
this fits the LG1 3 GPU / 2.5 CPU hour allocation. The CPU watchdog replaces
the generic four-times-wall reservation for this measured job; it is enforced
by the master. Neither timeout nor forecast changes the campaign allocation.
Execute `P2-sample` against the master-reference-01, controls-02 and tune-02
roots, writing a fresh `surrogate-hmc-all-models-20260919-lg1-sample-01` root.
Fresh tuning and retained sampling share a fixed kernel but no retained draws.
The six-function/two-look multiplicity family applies separately to each arm;
A is the exact-force diagnostic and does not select B's checks or continuation.

The retained schedule, when admitted, is four chains in250-transition chunks,
two looks at1000/2000. Six functions and margins are frozen in the claim reference:
standardized phi mean(.1), centered second moment(.1), reference .1/.5/.9 CDF
events(.02 each), and model-conditional Pr(y11>0|y)(.03). Compare with simultaneous
Student-t intervals at family alpha.05 over6functions*2looks, autocorrelation-aware
batch-mean MCSE with half/double batch-size sensitivity, plus the explicitly
empirical quadrature allowance. Require R-hat<=1.01, bulk/tail ESS>=400 and
adequate effective batches for every function. Ambiguous intervals or MCSE
diagnostics mean inconclusive, never equivalence. Preserve all arrays and seeds.
One chain group/data/bank cannot establish confirmation or general superiority.

Before retained results, fix batch lengths(n/40,n/20,n/10), use the maximum MCSE
and conservative36 within-chain degrees of freedom. Require largest/smallest MCSE
<=2, absolute pooled within-chain batch lag1<=.3 at the longest length, and
function-wise R-hat<=1.01. The lag check is conservative: even an IID realization
can be inconclusive. These checks are precision hypotheses, not guarantees of a
CLT. Independent IID, shifted-mean, stuck-chain and AR(1) long-run-variance tests
check the assessment code before use. Numerical diagnostics use TF graph mode
without XLA only where tied ranks are data-dependent; batch means are XLA.

Before usefulness claims, run the master’s cheap random-walk, prior-importance
and Laplace alternatives, with quadrature as the oracle. Evaluate nominal versus
tail regions and label runtime differences descriptive without replication.
Failure to complete those alternatives leaves practical usefulness unassessed;
it cannot erase or establish the separate B/R sampling-accuracy result.

Skeptical audit: the exact comparator and B share the actual observations, proper
prior and chart; independent references already checked their stationary law.
Separate A/B tuning prevents transferring a kernel across fields. Existing P1
tests establish the full call chain, replay, row independence and precision
semantics at the selected controls (selection chose the original multiplier1).
The limited L grid and short verification can fail through tuning rather than
the idea; preserve that distinction. No posterior result has selected this plan.
Reference/target corruption or invalid maps halt dependent work; ordinary
candidate failures trigger bounded repairs. Artifacts use new
`surrogate-hmc-all-models-20260919-lg1-*` directories, master manifests/accounting,
and a result note with the four comparison statuses and inference-status table.
