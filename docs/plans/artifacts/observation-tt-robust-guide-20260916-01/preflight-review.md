# A10 preflight review, 2026-09-16

Skeptical executor review PASS for the smoke and bounded comparison protocol.
The proposal preserves the exact particle importance target, evaluates both
mixture densities at selected draws, and transforms the physical initializer
rather than substituting chart covariance. Baseline remains A09 degree4 warm
TT. Repaired-guide-only, stable-chart and physical-defense arms isolate costs.
Guide resolution does not certify quadrature accuracy; real references remain
required. Calibration cannot exclude hard sequences or maximize MSE over the
defense weights. Failed calibration may freeze a diagnostic representative but
never turns it into an admissible candidate. Complete coverage, evidence and
conditional heuristic screens govern promotion. References use stride1000;
seed blocks and actual consumer call chains have executable tests.

MathDevMCP dispositions are in math-review.md. Eleven focused numerical tests
and eleven related regression tests passed. Expanded call-chain/seed/selection
checks are running before GPU launch. CPU-only tests intentionally hide GPU.
No scientific claim is made from the smoke; it uses no accuracy reference.

Exact smoke: CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true
/home/chakwong/anaconda3/envs/tftwogpu/bin/python
 docs/benchmarks/run_observation_tt_robust_guide.py --stage smoke
 --output-root docs/benchmarks/artifacts/observation_tt_robust_guide_20260916/attempt-smoke-01
 --wall-budget-seconds 600

Escalated GPU/XLA float64 with verified memory growth is required. All output
is logged by the driver. Calibration/confirmation use the same command with
fresh stage-specific roots, --wall-budget-seconds 5400, and confirmation adds
--calibration-root. Source hashes are checked for drift at attempt closeout.
A10 began 12:48 UTC; total ceiling20:48 UTC, numerical cap18000 seconds.
No independent-agent review claimed or launched.
