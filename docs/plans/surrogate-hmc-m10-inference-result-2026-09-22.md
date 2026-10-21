# M10 surrogate-HMC B/R result

M10 completed public HMC tuning and2,000 retained draws per chain across four
chains. All eight retained chunks passed numerical-health checks. B/R is
INCONCLUSIVE: the first look had valid uncertainty but insufficient precision;
the second failed the frozen Monte Carlo uncertainty screen for the log-Q
median-CDF functional. This is an actual test of endpoint-corrected surrogate
HMC with an imperfect full canonical analytical DCTR force. It establishes
neither posterior equivalence nor a posterior discrepancy.

## Target and executed computation

The unchanged M10 sine-transition, linear-plus-quadratic-observation model has
T=2, N=K=1024 and the original nominal claim observations. Infer log Q/R
multipliers under independent Normal(0,log(2)^2) priors in identity coordinates.
The proposal uses full FP32/TF32 canonical DCTR with the independently selected
protection multiplier0.1; integration is FP64. Acceptance uses the frozen FP64
log-domain32/128 quadrature endpoint and full prior. The independent R is
posterior05, with both observation roots, empirical refinement allowances and
analytical parameter-tail bounds. This numerical endpoint approximates the
model likelihood; it is not an exact likelihood oracle or the finite DCTR
likelihood. No force/endpoint gradient equality is asserted or required.

The [frozen inference plan](surrogate-hmc-m10-inference-2026-09-22.md) and
[endpoint qualification](surrogate-hmc-m10-endpoint-result-2026-09-22.md)
specify the13 functions, four comparison families, margins and only two
retained looks. Warmup and fresh tuning verification were archived and
excluded from inference. The public verified tuner passed all12 checks,
selected L=1 and step size1.1362549819623575, and produced an independently
verified tuning artifact. Its dense metric covariance was
`[[.381633909,-.101556817],[-.101556817,.273013279]]`.

## Findings and decision

At1,000 draws/chain all13 functional MCSE checks passed, but the simultaneous
intervals did not establish equivalence or discrepancy. At2,000, the largest
parameter R-hat was1.000765526, minimum bulk ESS2213.419 and minimum tail
ESS2556.397. These diagnostics are descriptive and do not override the
functional uncertainty veto. The log-Q median-CDF batch lag correlation was
-.347648884, outside the predeclared absolute limit.3. Its MCSE screen failed;
the remaining12 functional screens passed. No function had a discrepancy
interval, but invalid final uncertainty prevents interpreting that absence as
agreement. Do not relax the screen, repeat a closed look or re-score these
draws under a new rule.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve B/R INCONCLUSIVE | Simultaneous posterior agreement not established | No retained numerical-health veto; final functional MCSE veto | First look underpowered; final log-Q CDF uncertainty invalid | Continue another ready family and preserve this closed attempt; finite-D qualification is a separate M10 task | No posterior equivalence/discrepancy or rejection of surrogate HMC |
| C/D, D/R, C/R NOT_RUN | Finite-target endpoint/reference missing | Missing prerequisite | Finite-likelihood distortion and C sampling remain unknown | Independently qualify D before those comparisons | B/R cannot establish practical finite-DCTR inference |

| Inference status | Finding |
|---|---|
| Hard veto screen | All retained states, endpoint status, movement and energy checks passed; one functional MCSE screen failed at the final look |
| Statistically supported ranking | None |
| Descriptive-only differences | R-hat, ESS, step size, runtime and individual posterior deviations |
| Default-readiness | Not established; heuristic comparisons NOT_RUN |
| Next evidence needed | Prospectively specified independent data/banks and valid uncertainty; separate finite-D reference and C/D, D/R, C/R; conditional heuristic comparisons |

## Reproducibility and cost

The actual master command was
`scripts/run_surrogate_hmc_program.py --stage P3-M10-inference --scope M10 --output docs/plans/artifacts/surrogate-hmc-all-models-20260922-m10-inference-02`
under `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, with escalated GPU1
UUID GPU-68251639-fe82-8f81-3ccc-2953c32e805b, memory growth, TF32 and XLA.
CPU thread limits were two; Git baseline was8a4cffa2fd085f43ab5710c24d6e7b48770aea0d.
The manifest archives the exact dirty source closure, environment, seed
schedule, input hashes and commands. Peak TensorFlow GPU allocation was
68,799,744bytes. No source drift or resource-watchdog termination occurred.

The [worker report](artifacts/surrogate-hmc-all-models-20260922-m10-inference-02/P3-M10-inference/report.json),
[run manifest](artifacts/surrogate-hmc-all-models-20260922-m10-inference-02/P3-M10-inference/run-manifest.json),
[accounting](artifacts/surrogate-hmc-all-models-20260922-m10-inference-02/accounting.json)
and [master decision](artifacts/surrogate-hmc-all-models-20260922-m10-inference-02/decision.json)
preserve the result. The master reader revalidated source/data, frozen inputs,
tuning traces, sample/continuation hashes and closed-look rules against the
real archive: artifact_valid=true, closed=true, authorizes_new_inference=false.
There were59 passing focused checks before launch. The worker exited0; master
exit2 correctly records the incomplete all-model implementation.

The master charge was1,277.057653 CPU-worker seconds and1,252.217185 GPU seconds;
do not add the nested worker charge again. Global cumulative usage is
16,567.001321/54,000 CPU seconds and14,846.057197/226,800 GPU seconds. M10
remaining allocation is2,042.751206 CPU/5,128.549809 GPU seconds. Launch01's
pre-worker discovery failure and its cost remain preserved separately; it
consumed no tuning or statistical look.

Post-run red team: small parameter R-hat and healthy trajectories could tempt
a false success claim. The frozen functional uncertainty test blocks that
claim. The strongest alternative explanation for the inconclusive result is
limited precision plus an unstable finite-batch correlation screen, rather
than a biased proposal or failed target correction. The weakest evidence is
one short dataset and one bank under empirical quadrature allowances. A fresh,
prospectively designed experiment with valid uncertainty could establish
agreement or discrepancy; these archived results cannot decide between them.
