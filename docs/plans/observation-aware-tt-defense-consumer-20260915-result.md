# A05 result: smaller defense passes its safety bounds; SGQF remains necessary

The optional fraction **1e-5** passes the declared rescue/non-harm bounds and
all compiled sampler checks. The inherited 5% fraction fails the healthy-case
bound. TT still loses to exact SGQF in some first-transition cases; reducing
defense does not establish reliable fitting. Independent filtering remains
unmeasured for this candidate.

## Computation and evidence

[A05](observation-aware-tt-master-amendment-05-defense-consumer-20260915.md)
was independently reviewed AGREE before implementation. One research attempt
completed in 19.103 seconds on RTX 5080, float64 GPU/XLA, with verified memory
growth. It contains 96 stress configurations and 24 exposed-target cases with
fresh 8192-row designs. No research retry occurred. Artifacts are in
`../benchmarks/artifacts/observation_tt_defense_consumer_20260915/attempt-01/`;
run_manifest.json records command, environment, hashes, seeds and timing.

The four new consumer checks pass. Fourteen existing pair/initializer checks
passed in the first invocation. Its one failure was a test harness missing
the CLI's deferred TensorFlow/lib module bindings; supplying those bindings
made all four consumer checks pass. Logs preserve both invocations. No
mathematical implementation was changed to repair that test.

The optional Gaussian consumer represents exactly the specified approximate
joint q_t(x)q(z|x), with analytically conditioned x|z and the exact q_t retained
marginal. Independent tests verify joint-minus-marginal log density, the
linear-Gaussian exact limit, sample/evaluated-density parity and a real particle
filter endpoint mixing Gaussian and TT steps. It is not a proof that the SGQF
closure equals the nonlinear filter. No runtime selection policy is promoted.

## Dedicated safety result

For N=512, healthy c/Z>=1/4 and stress c/Z<=sqrt(eps64), the admissible interval
derived before observing target errors is [7.6144e-6, 4.88998e-4]. The smallest
passing grid entry is 1e-5; 1e-4 also passes. At 1e-5 the healthy boundary has
0.02048 expected changed draws and the stress boundary 0.76180 expected
unprotected draws. At 5% the healthy value is 89.04348, above the declared
one-draw budget. The selected value is frozen before target diagnostics.

Conditional densities, zero-mass rescue, amplitude scaling by 1e-4/1e4 and
common-random-number polynomial branch identity pass. Maximum direct-density
disagreement is 1.43e-14. The conditional fraction is ancestor-dependent, and
its extrema/draw count are now retained in sampler diagnostics. These bounds
address near-zero conditional mass; they do not certify nonlinear tail coverage.

Median empirical Hellinger squared discrepancy on fresh numerical rows:

| d, time, incoming law | Exact SGQF | SGQF-start TT, zero defense | TT, 1e-5 | TT, 5% |
| --- | ---: | ---: | ---: | ---: |
| 1, 1, Gaussian | .001234 | .000591 | .000591 | .000902 |
| 1, 1, recursive TT | .002377 | .000612 | .000612 | .000881 |
| 1, 17, recursive TT | .003391 | .000850 | .000850 | .000946 |
| 1, 18, recursive TT | .005336 | .000779 | .000779 | .000867 |
| 4, 1, Gaussian | .001949 | .003111 | .003111 | .005147 |
| 4, 1, recursive TT | .005820 | .004813 | .004813 | .006920 |
| 4, 17, recursive TT | .009953 | .003557 | .003557 | .004885 |
| 4, 18, recursive TT | .217442 | .120002 | .120001 | .121747 |

Both start families and the product Gaussian comparator are preserved in
summary.json and per-case files. Eight of 48 family/case comparisons at 1e-5
lose to SGQF, which is an empirical promotion veto. These numbers are descriptive
and did not select the safety fraction. At d4 time 18, target row ESS falls to
11.94 of 8192 and the largest single normalized target weight is .28690.
Evaluation coverage is inadequate for a precise ranking there. The different
ordering of generic and warm fits from A04 reinforces that uncertainty.

## Decisions and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit 1e-5 as optional bounded safety candidate | Derived bounds pass | Sampler checks pass | Chosen stress/loss budgets and real tails | Freeze only within next reviewed protocol | Universal default or full tail protection |
| Preserve exact SGQF consumer | Conditional algebra and endpoint pass | No density/SPD failure | Actual nonlinear filter accuracy | Include as explicit comparator and fallback | Exact nonlinear filter |
| Do not promote TT | No independent filtering result | Some early cases lose to SGQF | Late row concentration | Frozen-candidate independent-sequence comparison | Reliable fit, superiority, HMC |

| Inference status | Result |
| --- | --- |
| Hard veto screen | No numerical validity veto; 5% fails declared non-harm; TT has heuristic promotion veto |
| Statistically supported ranking | None |
| Descriptive-only differences | All target-error means/medians and row-design comparisons |
| Default readiness | No change; candidate remains optional |
| Next evidence needed | Independent sequence-level filtering, accurate references and paired uncertainty |

Post-run red team: synthetic protection may pass while actual target tails are
missed. The d4 late weight concentration is direct evidence for that concern.
A well-covered independent comparison could overturn the current apparent fit
ordering. The weakest part of this evidence is population error estimation,
not the checked Gaussian conditioning or mixture identities. Proceed to the
master's phase 9 with a frozen fitting/selection policy and untouched sequences;
do not tune the defensive fraction using this table.
