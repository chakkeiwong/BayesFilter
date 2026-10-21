# LG2 scoped protection calibration and inference-cost forecast

The main question remains whether imperfect full canonical DCTR forces support
endpoint-corrected HMC parameter inference. This stage asks whether an LG2
protection setting can be selected without materially changing healthy
computations, and whether subsequent tuning and retained inference fit the
existing allocation. It is not a posterior comparison. The qualified matrix
Kalman endpoint, data law and parameter chart are unchanged.

Use attempt 1, replicate 0, T=20, N=K=1024 and the separate calibration and
validation partitions in `surrogate-hmc-all-models-20260920-lg2-reference-01/reference`.
Never read claim observations or posterior functions during control selection.
The actual four-chain consumer passed all fourteen GPU qualification checks.
The scope uses full Contract E, analytical recursion, GenUT, UKF covariance and
dual-cap correction, FP32/TF32 GPU/XLA with verified memory growth.

Compare protection multipliers 1, .1, .01, .001. Their shared construction is
`surrogate_hmc_dctr_selection.protection_controls`: ridge 1e-5*m, damping
1e-3*m, scale floor 1e-6*m, trust radius .1/m, pairwise cap 2/m and coordinate
cap 16/m. This is an inherited calibration hypothesis, not an LG2 default.
The remaining substep, transport and correction settings are explicitly frozen
warm-start hypotheses. Their adequacy is not established by the protection
curve. No control is tuned against posterior comparisons or heuristic outcomes.

Evaluate nominal physical rows (.65,-.4), (.85,0), (1,.4), (1.1,-.2), spanning
transition persistence and process scale. Evaluate stress rows at the four
corners (.5007 or 1.1993, -.998 or .998), one thousandth of each prior width
inside the boundary. Boundary stress tests can expose numerical failures hidden
at the truth. These fixed positions are convenience coverage, not posterior
coverage or a guarantee over the continuous prior rectangle.

For each adjacent pair, all nominal rows must retain valid original value and
force status and finite, bounded full filter/reset moments. The maximum change
in value, force and each moment tensor, normalized componentwise by
`1+abs(weaker)`, must be at most .001 on nominal rows. This inherited tolerance
is a non-harm hypothesis in the fixture's fixed units, not a general scientific
error bound. Each stress row healthy under the weaker setting must remain
healthy under the stronger setting. Other failed stress rows must emit original
failure codes. The gross 1e6 bound is an engineering rejection screen; no
computed quantity is clipped. Select the first adjacent pair that passes on
both partitions. The shared issuer must verify the declared model and attempt,
not a hard-coded LG1 attempt. A missing shared pair means NO_STABLE_CONTROL_PAIR
and triggers scoped repair, not rejection of surrogate HMC.

Preserve all original statuses, full moment checks, per-component differences,
data/source identities and settings. Exact matrix Kalman values and total
physical gradients are explanatory only. This control selection grants only
imperfect-proposal experimental use, with no exact-score or posterior admission.
The unchanged controls factory and canonical consumer do not need a second GPU
qualification solely because the issuer's model/attempt validation is repaired;
focused scope-rejection tests cover that nonnumerical change. Preserve the old
qualification snapshot, and freeze the new issuer in the selection artifact.

If controls are selected, measure one compiled four-chain, one-leap corrected
transition at step .02 and identity mass on calibration data: one compilation
and four timed transitions, seeds (20260920,301..305). This is the cheapest
permitted trajectory, not tuned HMC. Archive statuses and stop the timing probe
on non-finite state, force fallback or endpoint failure. Record process CPU and
synchronized wall time separately. An optimistic minimum forecast uses the
fastest observed CPU/wall transition times times 3250 transitions (250 fresh
warm-up, 1000 verification and 2000 retained), then a 1.5 cost allowance. It omits
nomination and its adaptation, A/C, finite-reference construction and recompilation; failure even
against this optimistic floor establishes under-budgeting for this allocation.
Passing it would still require a complete forecast and public tuning. Do not
claim a measured single-step cost is a statistical or hardware lower bound.

Budget: at most 900 GPU seconds / 1000 CPU-worker seconds for calibration and
the cost probe, plus 100 CPU-worker seconds of focused tests. These remain
within LG2's initial 7200 GPU / 1800 CPU-worker-second allocation (253.471058 CPU
seconds already spent before this stage); no total campaign expansion. Use the
master watchdog, one active GPU process and a fresh versioned output root.
Stop on invalid inputs, corrupted artifacts, resource limit or invariant
failure. A budget shortfall defers retained inference; it cannot justify shorter
unplanned chains or relaxed criteria. Give other ready families a turn before
another exploratory allocation, as the master requires.

Skeptical audit: transferring LG1's selected controls would fail scope validity;
this stage constructs LG2 evidence. Selecting on force accuracy would answer
the wrong question; only the non-harm curve selects. A successful calibration
cannot establish posterior recovery. Claim data remain untouched. The measured
cost forecast precedes serious tuning. The independent exact reference is the
comparator for later B/R; later C/D and D/R require their own finite target and
reference. Heuristic comparisons are still outstanding and veto promotion.

Pre-launch check: 50 focused orchestration, scope-selection, reference and
budget-screen tests passed in `surrogate-hmc-all-models-20260920-master-lg2-tests-02`.
The first test run exposed a stale single-scope test expectation after adding
multiple scope cards; it was corrected and rerun. Both runs remain charged.
The dry run selects only the LG2 calibration worker, its frozen reference and
the declared 900 GPU / 1000 CPU-second watchdog. GPU 1 was idle before launch.
No scientific criterion changed during this repair.

Execute the tftwogpu Python through `scripts/run_surrogate_hmc_program.py
--stage P3-LG2-controls --scope LG2 --reference-root <reference> --output <fresh>`
with escalated GPU access. Save the command, Git commit, environment, seeds,
wall/CPU time, GPU/memory policy, source snapshot, selection, curves, transition
traces, forecast, accounting and result under that output. No public tuning
artifact is issued here. Only `tune_hmc_kernel` with the typed position-field
binding and `VerifiedTensorFlowHMCConfig` can authorize later retained sampling.
