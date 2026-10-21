# AR2 posterior execution within the surrogate-HMC closure master

The question is whether canonical analytical DCTR can drive useful HMC
proposals for the declared AR2 model when independent model-likelihood
Metropolis acceptance corrects the endpoint. This is the existing master's
B/R experiment. Its comparator is the independently qualified Kalman/grid
model posterior. An A/R exact-gradient HMC comparator follows if actual B
costs leave its conservative forecast affordable. No Claude review is pending
or required for execution; the user deferred it.

The model and data are frozen in structural-03: two-dimensional state,
one-dimensional observation, T=10, F=[[.45,-.15],[1,0]], H=[1,0],
Q=.09 exp(2 lambda_Q) B B', B=[1,0]', R=.04 exp(2 lambda_R), initial
state N(0,I). The two log-scale parameters have independent N(0,log(2)^2)
priors. The held-out claim data use seed [20260923,503] and the declared
transition-first law. The canonical proposal uses N=64, FP32/TF32 and the
unchanged, source-bound validation-01 controls. Structural deterministic
filtering preserves the noiseless lag; no artificial process noise is added.
HMC coordinates, independent Kalman endpoint and reference are FP64.

The force is a frozen, position-only approximation to the model-posterior
potential gradient. The claimed stationary target is the model posterior
because the reversible, volume-preserving kick/drift proposal uses a separate
Metropolis ratio computed from that posterior and kinetic energy. The force
is not claimed equal to the exact gradient. Reversibility/volume preservation
follow from composing position-dependent momentum shears with momentum-dependent
position shears and the momentum reversal; gradient equality is unnecessary.
Floating-point implementation and diagnostic validity still require tests.
The deterministic finite likelihood is not this experiment's target.

## Evidence and stopping contract

Primary B/R evidence is the 13 previously specified functions: two standardized
means, three second moments, six marginal CDF probabilities and two predictive
probabilities. Margins are .1 for the first five and .02 for the last eight.
Use the scope-calibrated guarded-lugsail assessment at cumulative 4000 and 8000 retained draws
per chain, four chains, with the empirical reference integration allowance.
Every simultaneous interval must lie within its margin for EQUIVALENT. An
interval entirely outside its margin gives DISCREPANCY. Otherwise retain
INCONCLUSIVE. Qualification is empirical, not a rigorous integration theorem.

The diagnostic must first pass fresh AR2 calibration under
`surrogate-hmc-ar2-mcse-repair-2026-09-23.md`. The original assessment failed
its initial calibration and one fresh confirmation; neither qualifies this
run. Modern parameter and function R-hat, parameter
bulk/tail ESS, MCSE batch-size sensitivity and stationarity are the frozen
diagnostic vetoes; their exact thresholds are in the scoped uncertainty plan.
They are checked by the same callable used in calibration. Unqualified first
look means continue to the planned second look. Equivalent or discrepant first
look ends this candidate's retained stage. A discrepancy rejects this candidate,
not the research direction, and triggers the master's discrepancy localization.

Nonfinite states, invalid endpoints, failed archived provenance, excessive
energy error, all-chain movement failure, or any canonical-force fallback
during fresh verification or retained sampling veto the corresponding evidence
and trigger a localized repair. Never disguise a substituted force as canonical
DCTR. Adaptation fallback counts are retained as explanatory diagnostics;
adaptation samples cannot enter posterior summaries. No artifact corruption,
unavailable calibration or exhausted allocation may be bypassed.

Acceptance, runtime, force/gradient differences and raw likelihood offsets are
explanatory. Observed discrepancies inside the pilot margins are reported
separately from formal equivalence and owner acceptance. There is one primary
comparison family, B/R. A/R is a descriptive comparator and cannot enlarge that
family's formal claim. No efficiency ranking, finite-target fidelity,
default readiness or all-model validity follows from this run.

Cheap adversaries are already constructed: direct joint-Gaussian observation
density, Kalman evaluation, independent grid sampling, and exact-gradient HMC.
The first two passed likelihood-law parity; grid calibration evaluates known
dependence, offsets and drift. The posterior functions examine central moments,
both marginal tails and predictions separately. A/R receives its own tuning.
Conditional approximation error against the model oracle is the primary
test; no speed advantage is claimed. Until A runs, efficiency against the
cheaper Kalman sampler remains not checked, not a favorable comparison.

## Frozen tuning and default audit

The public `tune_hmc_kernel` with `VerifiedTensorFlowHMCConfig` is the registry's
artifact authority for the typed, raw-coordinate position-force binding.
The binding supplies canonical force and independent endpoint separately.
The same mechanics with the exact gradient give A; neither arm uses an
arbitrary-force fixed-transport tuner. A fresh public tuning artifact is
required before the public retained runner can execute.

| Choice | Provenance and justification | Failure mode and earliest check | Status |
|---|---|---|---|
| Prior-scale initial metric, 128/256 mass windows | Own-target adaptation after a model-specific prior-scale start | Poor geometry; fresh dispersed verification | Warm start hypothesis |
| .1 shrinkage to P0, zero additive jitter | C=.9 E+.1 P0 >= .1 P0 for PSD empirical E; inverse is bounded by 10 P0^-1 | Metric distortion; report spectrum and posterior diagnostics | Proposal regularization hypothesis |
| Eigenfloor 64 eps times prior variance; condition limit eps^-1/2 | Scale-relative numerical backup; prevents amplification beyond about half the FP64 digits | Backup firing or nonfinite factor; actual regularizer diagnostic | Numerical protection |
| Initial step .2; 256 adaptation draws, target acceptance .7 | Conservative normalized-coordinate hypothesis, then arm-specific public adaptation | Nomination rejects or fresh verification fails | Tuned, never transferred |
| L in 1,2,4,8; 128 nomination draws | Same complete-transition cost envelope as validation-02; first passing public candidate | Short trajectory mixes slowly; independent R-hat and retained ESS | Budgeted candidate ladder |
| Nomination overall acceptance [.5,1], per chain [.3,1] | Avoid nominating nearly immobile kernels; not convergence evidence | High acceptance hides slow exploration; fresh R-hat | Nomination screen only |
| Four starts at prior SD times (-1,-1),(-.33,.33),(.33,-.33),(1,1) | Dispersion in both parameters without using claim posterior draws | Missed remote regions; grid/CDF comparison | Initialization hypothesis |
| 500 fresh warmup, 2000 fresh verification | More than the public 1000-draw minimum; excluded from retained estimates | False apparent mixing; retained diagnostics remain required | Scope-specific bounded verification |
| Energy error bound 1000 | Existing severe numerical veto; far beyond useful acceptance | Smaller integration errors still harm movement; acceptance and ESS expose them | Veto, not tuning target |
| 250-draw chunks, cached compiled kernel | Identical frozen target/settings; state and seed are explicit inputs | Stale graph reuse; checksum, cache invalidation and input tests | Engineering repair |

The metric protection is evaluated before tuning using the actual public
regularizer: the healthy prior metric is unchanged and a rank-deficient metric
receives exactly the derived .1 P0 protection. Shrinkage intentionally changes
other empirical proposal metrics; endpoint acceptance preserves the target.
Actual posterior agreement is still required. No safety parameter is silently
selected on posterior-test data.

Tuning has at most 384+4*(256+128)+500+2000=4420 transitions per chain,
inside the measured 7000-transition overhead envelope. Verification is fresh;
all tuning/warmup archives are retained and excluded. Seed base [20260923,
22000] for B, [20260923,21000] for A; each subsequent output run index adds
10000 to the second component. Fresh verification adds 1, retained chunk
seeds add 100+index, and assessment seeds add 1000+retained_count. The public
runner also rejects reused continuation seeds.

## Budget, execution and skeptical audit

The inference pass reserves at most 21500 CPU-worker / 25000 GPU-wall seconds
from the unchanged global 54000 / 226800 campaign caps. B has 21000 CPU /
22500 wall seconds, with 200 CPU seconds for focused preflight. Measured
conservative B forecast is 20517.539 / 21407.652. A's separate forecast is
2968.836 / 3069.349; its worker allowance is 3200 CPU / 3400 wall seconds.
Run A only if B's actual charge leaves the forecast plus preflight affordable
in both campaign and pass budgets. No simultaneous full reservation of A+B
is assumed. Up to three attempts per inference arm remain bounded by the same
pass and global caps; changed tuning requires recorded rationale and fresh
archives, never tuning on failed claim samples.

Use the trusted tftwogpu environment, GPU memory growth and XLA. CPU-only
preflight is explicitly diagnostic. Entry command is
`scripts/run_surrogate_hmc_program.py --stage closure --closure-stages ar2-inference-b`
with a fresh `surrogate-hmc-closure-20260922-ar2-inference-NN` output root.
A uses `ar2-inference-a`. Preserve commands, source hashes, consumer identity,
data, tuning and chunk tensors, diagnostic looks, memory policy and accounting.
The master must distinguish inference launched from posterior test completed.

Skeptical audit passes provisionally subject to focused tests and calibration:
the model posterior is the correct baseline, the numerical proposal remains
canonical, and retained intervals rather than acceptance decide the stated
accuracy question. The worst remaining explanation for an inconclusive result
is inadequate mixing or precision, not model failure. The weakest external
assumption is transfer from the controlled diagnostic laws to actual chains;
the actual-chain checks and reference comparisons remain mandatory. No new
scientific direction or compute expansion is introduced.

## Current pre-run audit after validation-06

Validation-06 qualifies the frozen assessment method
`ar2_guarded_over_lugsail_r2_c_two_thirds_plus_r3_v2`. All six predeclared
regimes passed on 1,920 new trials. The refresh-.9 interval-failure Wilson
upper bounds are .040516 unconditional (12/512) and .038807 admitted
(12/502). Negative-control lower bounds are .970863. This qualifies the
uncertainty diagnostic under the tested laws; posterior validity still depends
on the actual B/R experiment. The full report is
`artifacts/surrogate-hmc-closure-20260922-ar2-validation-06/ar2-calibration-repair-r2/report.json`.

The execution review repaired the stale validation-05 inference pointer,
missing closure CLI stages, pass accounting and launch reporting. The public
binding remains position-only DCTR with a separate Kalman endpoint. The
compiled retained kernel keeps state and seed as inputs and revalidates the
tuning archive on each chunk; focused tests check continuation against a fresh
kernel and invalidation after the artifact changes. These repairs do not
change the target, force, retained looks, margins or assessment callable.

The generic subprocess ceiling remains four hours. The explicit B job has a
22,500-second ceiling to accommodate the measured 21,407.652-second forecast;
its 21,000 CPU-second allowance, 21,500/25,000 pass limits and global limits
remain enforced. An explicit override is required and recorded in its manifest.
Three attempts per arm remain possible only within these same budgets.

Skeptical audit: pass subject to the mandatory focused preflight. The baseline
is the frozen model posterior, force accuracy and acceptance remain explanatory,
and no candidate can pass by omitting the retained diagnostics or reference
comparison. The qualification source hashes are checked before tuning. The
budget supports B first, and A only if B's actual use leaves enough capacity.
The warm-start metric and short nomination have the already stated dispersion,
fresh-verification and retained-sampling checks. A failure triggers its planned
repair; it does not itself reject surrogate HMC.

Exact launch, using the trusted GPU environment with memory growth:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages ar2-inference-b --output docs/plans/artifacts/surrogate-hmc-closure-20260922-ar2-inference-01
```

Before this launch the cumulative charge is 32,207.100328 CPU and
30,236.225435 GPU seconds; remaining global capacity is 21,792.899672 CPU
and 196,563.774565 GPU seconds. The 21,500/25,000 pass reservation fits.
The CPU-only preflight intentionally hides GPU devices. The numerical worker
must verify GPU placement and memory growth before executing HMC.


## Crash-recovery preflight, 2026-09-23

The editor crash occurred before an AR2 inference worker launched. Process
inspection found no surviving surrogate-HMC worker. The sandbox helper now
fails before commands start (mountinfo path is not absolute), so trusted tool
permissions are required; this is an execution-environment issue.

The CPU-only focused preflight passed 69 tests after repairing a test-fixture
bootstrap leak: tests/conftest.py enables the custom-op preload, and the
dry-run fixture imported the package before the CLI could disable it. The
fixture now opts out before that import, as the real CLI already does.
No inference or diagnostic formula changed. The initial preflight had 68
passes and this one fixture failure. Full repaired log:
`/tmp/surrogate-hmc-ar2-recovery-preflight-01.log`; dry-run:
`/tmp/surrogate-hmc-ar2-recovery-dry-run-01.json`.

The dry run confirms review plus ar2-inference-b, a fresh inference-01
directory and the unchanged 21500 CPU / 25000 GPU pass reservation.
Git diff whitespace checks passed; HEAD and branch match the checkpoint.
Trusted nvidia-smi sees both GPUs. The numerical worker additionally
checks TensorFlow GPU placement and memory growth before any inference.

Skeptical audit passes: endpoint target, source-bound force, public tuner,
fresh verification, retained looks, reference comparison and budget limits
are wired and covered by the focused checks. Continue the existing master's
B/R experiment; this is neither a new research direction nor an all-model
completion claim.
