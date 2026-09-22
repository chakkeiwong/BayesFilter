# A05: defensive-mass calibration and an exact SGQF joint consumer

Date: 2026-09-15. Status: REVIEWED; implementation and execution authorized.
Independent review returned AGREE in
`artifacts/observation-tt-defense-consumer-20260915-01/review-01.txt`.
The reviewer did not inspect code. Its approximate square-root machine-epsilon
value is corrected here: sqrt(eps64)=1.4901161193847656e-8.
This executes the remaining diagnostic part of master phase 8 under H6's
existing five-hour authorization. A04 is closed. It changes no production
default and does not repeat A04 fitting to seek a different result.

## Question and amendment

Can a much smaller defensive component protect the conditional sampler at
negligible polynomial mass, while preserving its healthy distribution? Can a
filter consumer use the unconverted SGQF joint with the correct conditional
density and retained marginal? A04 found that converting and fitting SGQF can
discard accuracy, and left the inherited 5% defense unjustified. The exact
Gaussian must remain available if a subsequent validation rule selects it.

The previous specification is A03 phases 7–10 and A04. This amendment supplies
the dedicated safety calibration and consumer mechanics; phase 9 still requires
a frozen candidate and its independent-sequence protocol. Target, charts,
d=1,4, float64 GPU/XLA and total budget are unchanged. Gaussian conditioning,
mixture diagnostics and a tagged consumer are extensions, with no Zhao–Cui
source-faithfulness claim. There is no coupled-chart rewrite.

## Mathematical contract

Write Z=integral h(u,v)^2 rho(u)rho(v), c(v)=integral h(u,v)^2 rho(u).
For joint defensive fraction a, tau=a Z/(1-a). The conditional density is

    q_a(u|v) = (h(u,v)^2 + tau) rho(u) / (c(v)+tau).

Its Gaussian conditional fraction is e(v)=a/((1-a)c(v)/Z+a), not a.
For c(v)>0, q_a=(1-e)q_0+e rho. Coupling the mixture to the same polynomial
draw gives conditional total-variation distance at most e(v) at each fixed v.
The joint TV bound is a.
At c=0 and a>0 the conditional is exactly rho. a=0 there is undefined and
is a diagnostic comparator, never an eligible safe configuration.

For incoming SGQF z~N(m,P), x|z~N(Az,Q), define S=APA'+Q,
K=PA'S^-1, B=P-KSK'. The available SGQF joint is

    q_G(z,x)=N(x;m_t,P_t) N(z;m+K(x-Am),B).

Thus its z mean is b=m+K(m_t-Am), covariance R=B+KP_tK',
cross covariance Cov(x,z)=P_tK'. Its consumer draws

    x|z ~ N(m_t+P_tK' R^-1(z-b), P_t-P_tK' R^-1 K P_t).

The retained current density is exactly N(m_t,P_t). The log conditional must
equal log q_G(z,x)-log N(z;b,R), including normalizers. Initial time uses
q_t directly. Both Gaussian and TT variants expose their actual retained
current marginal and evaluated conditional density. Particle weights remain
g(y|x)f(x|z)/q(x|z). This is exact for the stated approximate joint, not proof
that SGQF equals the true nonlinear filter.

## Safety calibration, defaults and decision rule

The safety purpose here is rescue near vanishing conditional polynomial mass,
not coverage of every tail of the nonlinear target. Fix N=512, the planned
downstream particle count. A healthy boundary is c/Z>=1/4; a numerical-stress
boundary is c/Z<=sqrt(eps64). These are explicit calibration hypotheses:
the former is within a factor four of average conditional mass, and the latter
is a conservative small-mass stress scale. Neither classifies arbitrary
scientific tails as safe. Report actual c/Z distributions separately.

Require on the healthy boundary N*e<=1 (at most one changed draw in expectation
under the coupling), and on the stress boundary N*(1-e)<=1 (at most one
unprotected draw in expectation). This translates a stated particle-level
loss budget into bounds on a; it does not tune to target error. Test the fixed
grid [0,1e-8,1e-6,1e-5,1e-4,1e-3,.005,.01,.05]. Select the smallest positive
entry satisfying both bounds and finite, bracket/CDF and exact-density checks
for both d=1 and d=4. If none passes, no mass is admitted; record the conflict.
The choice is a bounded sampler-safety candidate, never a universal default.

Stress fixtures have h(v,u)=v_1 times a constant current amplitude, with
v_1^2 in [0,eps64,sqrt(eps64),1/4,1,4], Z=1. This permits exact conditional
Gaussian density, including the c=0 branch, and tests the real compiled sampler.
Also use a nonconstant current polynomial with an exact squared-Hermite law
to check common-random-number identity when both draws use the TT component.
Scale all amplitudes by 1e-4 and 1e4 to check scale invariance. These synthetic
cases isolate validity; they do not establish nonlinear coverage.

Evaluate frozen A04 selected generic and SGQF-start cores on fresh numerical
rows (seed 915100+1000*d+10*t+r, r=0,1,2; 8192 rows each) for exposed Gaussian
incoming t1 and actual recursive t1,t17,t18. Keep A04 cores/L1/schedules fixed.
Record c/Z, conditional e, rho/q bounds, and target Hellinger discrepancy for
a=0, the admitted value, and .05. Target errors, importance concentration,
Gaussian comparison and row-design variability are explanatory only; they
cannot select a. Fresh rows do not make exposed observations a holdout.

The cheap adversaries for this diagnostic are the exact SGQF joint, the product
Gaussian reference, the undefended fitted TT and the inherited 5% TT. Report
them separately at first Gaussian transition, first recursive transition,
ordinary late recursive time 17 and stressed time 18. Any per-case Hellinger
loss to SGQF vetoes
TT promotion; it does not veto testing the Gaussian consumer or a later
predeclared safeguard. No superiority claim follows from three row designs.

The fixed degree 3/rank 3, A04 L1 selection and exposed incoming densities are
baselines with known limitations. No refit occurs here. The Gaussian covariance
algebra uses checked symmetric factorizations, without a silent ridge. An SPD
failure is a validity veto. The scalar grid, stress scale and particle loss
budgets are declared hypotheses; sensitivity tables preserve their limitations.

## Evidence roles, checks and stops

Primary pass: the derived non-harm/rescue bounds and valid consumer mechanics.
Promotion veto: failure of any bound, conditional density identity, finite/SPD,
CDF residual <=1e-8, bracket, scale invariance or actual endpoint wiring check.
Continuation veto: wrong target/density, corrupt or missing artifacts,
source/fixture drift, unrepairable numerical invalidity, or exhausted budget.
A failed mass or a TT loss to SGQF rejects that candidate only. Routine localized
repairs preserve attempts and the unchanged scientific contract.

Focused tests compare SGQF conditionals with an independently formed joint
Gaussian, verify the linear-Gaussian exact limit, sample/log-density parity and
retained marginal, and exercise a mixed Gaussian/TT consumer endpoint. Existing
pair tests remain relevant. Tests deliberately hide GPUs; the diagnostic uses
trusted GPU/XLA, stable signatures and verified memory growth. One-time Gaussian
setup, saved-result reading and post-run statistics are explicit TF/stdlib
reference/reporting exceptions, not a replacement numerical backend.

Pre-mortem: a passes synthetic bounds while real target tails remain missed;
therefore the real c/Z/weight diagnostics and later independent filtering are
essential and synthetic passing cannot promote a filter. A Gaussian sampler
may appear correct while returning a wrong conditional normalizer; joint-minus-
marginal identity and particle-weight endpoint tests address that risk.

## Budget, commands and records

Balance at resumption: 10996.896 seconds, with idle time excluded, in
`artifacts/observation-tt-sgqf-initialization-20260915-01/budget.json`.
A05 allows at most 3600 seconds of active work, including this design, review,
implementation, tests and interpretation; at most three numerical attempts and
900 seconds total numerical time, with 300 seconds per attempt. Reserve at
least 1800 seconds of the overall remaining budget for phase-10 records and
manuscript. No extra A04 allowance or new five-hour grant is assumed.

Commands after review and implementation:

    CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/highdim/test_sgqf_joint_consumer.py tests/highdim/test_sgqf_pair_initialization.py tests/highdim/test_pair_block_tt_remedy.py

Pre-execution skeptical audit: no baseline substitution or proxy promotion was
found. The 5% baseline is preserved, the exact SGQF conditional comparator is
explicit, and exposed observations cannot become independent evidence. The
main unresolved assumption is the safety loss budget, hence optional status
and explicit sensitivity reporting. The six-arm filtering ladder is deferred
until a frozen candidate; this bounded diagnostic does not answer it.
    bash docs/plans/artifacts/observation-tt-defense-consumer-20260915-01/run_diagnostic.sh attempt-01

The wrapper selects RTX 5080 by UUID, enables memory growth before import and
runs `docs/benchmarks/diagnose_observation_tt_defense_consumer.py` with a unique
output root under `docs/benchmarks/artifacts/observation_tt_defense_consumer_20260915/`.
The manifest records Git commit/dirty state and source/input SHA-256, exact
command, environment, GPU/TF32/XLA/growth, seeds, all elapsed time, plan and result.
No prior result is overwritten. The result note contains decision/inference
tables, actual-versus-stated target, red-team interpretation and the next
justified phase. Update master/checkpoint before execution and at closeout.
