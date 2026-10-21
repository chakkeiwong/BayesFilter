# M13 failed fresh-start localization

This is the force/initialization repair already included in the authorized
surrogate-HMC master. The target, data1704/T20, selected FP32/TF32 analytical
DCTR force, frozen FP64 endpoint, and posterior criteria remain unchanged.

Public tuning03 nominated epsilon 0.310948 and L=1. The fresh chain starting at
(-log(2), -log(2)) accepted zero of 1500 transitions and flagged fallback at
each transition. The other three chains had no fallback. Thus the aggregate
fallback count does not mean every chain failed. Energy errors above 2000 in
the stuck chain make this a genuine failed tuning candidate. Its endpoint
remained finite; it is not evidence that surrogate HMC as a class is invalid.

First inspect the four actual starting forces, their validity, the nominated
metric and step size, and deterministic proposal trajectories at those starts.
Compare starting behavior with four fixed posterior-scale starts obtained from
the already archived RWM warmup covariance and center, never retained draws.
Use the same force/endpoint call chain; do not introduce a replacement field.
The cheapest discriminating test is one force evaluation at each original
start followed by a small fixed-step trace with the public tensor kernel.

Diagnostic roles: nonfinite calculations or invalid source/data identity stop
this diagnostic. Large finite force, poor energy conservation, rejection, or
fallback explain the rejected candidate and may motivate a bounded step or
initialization repair. They cannot promote a kernel. Promotion still requires
the public tuner, fresh four-chain verification and independent retained
posterior assessment. No speed ranking, exact-endpoint-gradient claim, or
all-model conclusion follows.

Budget: at most 600 CPU-worker and 600 GPU seconds for this localization,
charged to the existing 10000/10000 repair reserve. The materiality repair used
1386.822697 CPU / 1266.589424 GPU seconds. RWM resume06 used 2084.813698 CPU;
launch-only resume02--05 add 12.513541 CPU. Remaining repair reserve before this
diagnostic is 6515.850064 CPU / 8733.410576 GPU seconds. Each job must record
actual charges; no automatic expansion is permitted. Artifact root:
`docs/plans/artifacts/surrogate-hmc-m13-force-localization-20260927-01`.

Assumptions and skeptical audit: prior-scale starts were a tuning hypothesis,
not evidence of suitable posterior geometry. A large force there could be a
valid but poor proposal field, or a numerical fault. Record both the raw force
and validity flags before choosing a repair. Exact endpoint values are the MH
authority. Shrinking starts alone must not be presented as repairing a wrong
force calculation or establishing global convergence. Controls remain frozen;
changing them requires fresh scope-specific selection. No full retuning or
posterior run is authorized by a successful diagnostic alone; those are already
separate stages of the master and retain their declared budgets and gates.
