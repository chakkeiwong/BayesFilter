# Oracle initialization and population Gaussian controls

Status: COMPLETE; 25 terminal checks pass. [Result](artifacts/iapf-oracle-start-population-20260922-01/result.md).
Skeptical review PASS. This continues the authorized 48 CPU /
48 GPU hour campaign after the representation diagnosis. No production edit.

Question: would a correct initial diagonal guide survive the existing fitting
criterion, and does concentration persist under an ideal known Gaussian sampling
law? The previous phase identified cross-term bias and weak sampled curvature,
but did not show whether fixing initialization alone is enough or whether the
population fitting measure can itself hide global error.

Part A uses all 64 unchanged phase8 problems, with the same original box,
2000-step cap, realized tolerance and R L-BFGS-B controls. Change only the start
to the saved KL-optimal diagonal oracle, already verified inside all 16 boxes.
These are oracle-start diagnostics, not implementable fits or new paper results.
Compare previous R fits, original TF fits, oracle start and final fit,
conditionally by dimension/time/actual versus exact target and objective.
Require initial objective/analytic-gradient identities against the saved phase7
oracle state, and the same projected-gradient convergence gate as phase8.
Nonconvergence consumes budget and remains visible; it is a candidate failure,
not a continuation veto. Measure delta-KL from the oracle and objective/shape
changes without treating KL as the fitting objective or any movement as an
algorithm failure. An exact-target fit that moves from the KL minimum to large
KL while reducing its stated loss makes initialization-only repair inadequate
on that case. No numerical threshold or default is selected from this exercise.

Part B uses exact Gaussian integration, plus bounded independent R draws, to
separate sampled fitting geometry from global guide geometry. For a known
target b(x)=exp(-||x||²/2), proposal q=N(0,r I), and a guide with the same
covariance shifted by delta in coordinate one, define weights w=b². Then
E_q b^k=(1+k r)^(-d/2), and asymptotic ESS/N=(sqrt(1+4r)/(1+2r))^d.
The population normalized shape error is
S_q=1-exp[-delta² r/(1+2r)], whereas KL[target || guide]=delta²/2.
Derive these by completing the square. This is a known controlled law, not a
claim about the distribution of captured iAPF clouds. The ESS expression is a
large-N limit; it is not a finite-N prediction and cannot fall below one as a
claimed actual ESS. The ratio estimator's finite-sample bias is preserved.

Use dimensions 5,10,20,40,80 from GJL section5.2, plus d2 as a small control;
r=1 is an exactly matched Gaussian sampling law, r=0.5 a declared narrower
Gaussian hypothesis, and r=1/d a dimension-scaled concentration control.
None is an inferred original-author choice. Use delta=0 and3 (KL0 and4.5).
The narrow-law limit additionally uses r=1e-2,1e-4,1e-6 analytically, to verify
that small population shape error need not imply small KL. This is a proof
counterexample, not a proposed sample-generation scheme.

Independent draws: N=1000 (paper count), 32 declared seeds 9101:9132 for each
dimension/r cell. Reuse each cloud across delta0/3 for pairing. Preserve all
per-seed ESS, max weight, sampled shape and analytic population values. Report
means with t-based 99% intervals as descriptive Monte Carlo summaries only;
do not infer a ranking or require the biased ratio estimator to equal its
asymptotic formula. Exact product-of-Gaussians formulas must match the specialized
isotropic expressions to 1e-10 and identical-guide S must be <=1e-12. Finite
accepted values, dimensions, weight normalization and source identity are vetoes.
Poor ESS/conditioning or discrepancy from a large-N limit is explanatory only.

Constructed comparators: exact full guide (identity check), KL-optimal diagonal
guide, saved cloud/QR initializations, and previous density/shape minima.
They diagnose fitting and global Gaussian geometry. Prior downstream Kalman,
bootstrap, constant-guide and one-step heuristic vetoes remain; none is cleared
by this phase. Promotion criterion: none for a filter. Primary diagnostic pass:
unchanged arithmetic identities, exact integration checks, complete conditional
records and honest separation of empirical loss, population loss and KL.

Default audit: starts are analytical controls; optimizer settings and bounds
are unchanged hypotheses with existing convergence diagnostics. Native density
scaling is the original declared objective, not assumed safe. The Gaussian
sampling laws are deliberately known mathematical controls, not cloud-law
estimates. N1000 is paper-anchored; seed count32 is a bounded explanatory batch,
not powered validation of filter superiority. No new optimizer tuning, learned
default, author identity, paper replication, TF32, model-score, LEDH or HMC claim.

Skeptical review PASS: a blind bigger cloud or another solver would confound
initialization, objective and sampling law. This plan changes one fitting input
and separately uses a fully known Gaussian law. It does not equate diagonal KL
optimality with Equation15 optimality, assert finite-N ESS from an asymptotic
formula, or attribute the actual clouds' behavior to an assumed Gaussian law.
Expected failures answer the question and do not stop the remaining controls.

Budget: transfer exact phase9 ledger; about45.850 CPU /47.803 GPU hours remain.
This phase caps CPU worker time at one hour, no GPU, up to four launches each
<=600seconds including repairs. CPU/base-R4.1.2, GPUs hidden, one BLAS/OMP thread.
Root `docs/plans/artifacts/iapf-oracle-start-population-20260922-01/`.
Command `python3 docs/benchmarks/diagnose_iapf_oracle_start.py --attempt <unique>`.
Preserve source/input hashes, exact commands, seeds, versions, output paths and
wall time; terminal decision and inference tables, skeptical review and remaining
budget. No new dependencies. Stop for changed input, broken identities, invalid
accepted numerics or budget; automatically repair bounded harness failures.
