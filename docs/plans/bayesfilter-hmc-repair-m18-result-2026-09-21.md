# M18 final integration, numerical parity and official guide

M18 is complete. The checkpoint specialization preserves the complete compared
numerical procedure on both GPU fixtures, the affected regression selection has
2090 named passes and two documented skips, and the updated official book is
installed as `docs/main.pdf`. The terminal audit has no invalid artifacts or
outstanding workers. M13--M18 bounded execution is complete; the remaining
scientific validation questions are listed at the start of the master program.
Closure occurred on September 22 in Asia/Shanghai; the September 21 phase name
and paths are retained for continuity.

Design: [M18](bayesfilter-hmc-repair-m18-design-2026-09-21.md).
Execution and repairs: [engineering note](bayesfilter-hmc-repair-m18-engineering-2026-09-21.md).
Evidence root: `artifacts/hmc-repair-master-2026-09-16/m18-r1/`.
The authoritative terminal records are `reconciliation-terminal.json`,
`parity-comparison.json`, `current-source-reconciliation.json`,
`final-test-inventory-r2.json`, and `guide-r3/install-review.json`.

## Implementation and numerical checks

The checkpoint writer now omits redundant arbitrary-object normalization only
for execution specifications, evidence and chunks already copied into JSON
types. It still rehashes their live contents at every existing check. General
candidate identity hashing, persistence cadence, corruption rejection and
scientific decisions are unchanged. The saved-input study checked 3160 records
containing 196475293 serialized bytes with exact hash equality. Five alternating
host repeats had median times 0.895736 and 0.670019 seconds for generic and native
hashing. Those are descriptive host measurements, not an end-to-end speed claim.

The regression audit also found a real lazy public-export defect: resolving
`admit_hmc_uncertainty_nomination_for_confirmation` imported unrelated numerical
modules and TensorFlow. Its direct lazy routing now returns the same callable
without that import. This was an implementation repair, distinct from the
obsolete single-winner and callback assumptions corrected in several fixtures.

Both GPU arms use the identical frozen source, path, target, starts, geometry,
search and seeds. The only intervention is the diagnostic hash-function hook.
Each search pauses after two work items, resumes, preserves all verified
members, selects the first identity before further sampling, exports/reloads
that member, discards 64 transitions and retains 128 per chain on fresh streams.
No posterior accuracy claim is made for these small fixtures.

| Target | Candidates per arm | Verified per arm | Work items per arm | Numerical comparison |
| --- | ---: | ---: | ---: | --- |
| Gaussian | 8 | 1 | 13 | Exact equality |
| Beta-binomial | 15 | 2 | 24 | Exact equality |

The comparison includes binding/scope, candidate settings and states, verified
membership, work order, initial states, seeds, samples, transition traces,
analyses, retained-chain states and retained traces. Only timing/device-runtime
metadata, paths and hashes containing those fields are excluded. The audit
independently reconstructs the projections from full original evidence and
checks 74 evidence records, 84 saved chunks, 1846 inline tensor checksums and
eight standalone tensor files. Every check passes. Both arms execute trusted
TensorFlow/TFP GPU/XLA with memory growth verified before device initialization.

Current source identity is
`bb4090ef90025ce9441a7dfb1b6d3e3a2116258b5c9416dfff291536f56606ac`,
based on Git `d86dadf68ea57772642c6802990f46a3c6a04c30` plus the recorded working
tree. All 488 current package files match the final frozen snapshot. Relative
to M16/M17, only checkpoint hashing and the lazy export differ. Preparation,
targets, transitions and posterior diagnostics are byte-identical. That fact
and the two-model intervention comparison support this scoped engineering
change; they do not relabel earlier model campaigns as current-source runs.
The prepared `source-r1` remains preserved but was not numerically executed.

## Regression evidence and its limits

The current affected selection contains 2092 tests: 2090 pass, two skip, none
are missing and none have unresolved failures. It includes HMC, fixed-transport,
neural-force, dense-IAF and inference-validation mechanisms and public-path
integrations. CPU reference checks deliberately hide GPUs. The two skips are
a tiny historical bootstrap whose repair budget is insufficient, and the
explicitly GPU-only operational-warmup test in this CPU batch. Actual GPU
preparation and warmup were exercised separately in the numerical phases.

Obsolete fixed-transport callback fixtures now call the explicitly historical
scheduler. Two public Gaussian oracles now test complete candidate sets,
identity-based member selection, durable export/reload, and fresh posterior
draws against analytic means/covariances. The dense affine value/Jacobian oracle
is preserved. Historical efficiency-selector fixtures remain explicitly
historical. Other repairs update an extracted-module monkeypatch, avoid an
unintended floating-point boundary in grid/cache tests, preserve the original
uncertainty candidate flag, and make its tamper test actually change the flag.
No runtime admission threshold was relaxed to make these tests pass.

The first inventory's any-pass precedence was wrong as a general reconciliation
rule. It is preserved as superseded. The final inventory gives precedence to
the latest complete named outcome, including failures and skips. Two old
oracle failures came from a batch started before their repair but completed
afterward. A final focused rerun, after all earlier batches ended, passed both
on the unchanged final source. Its preceding whole-module attempt hit a
120-second timeout; anonymous progress dots are not passing evidence, and the
attempt is charged conservatively at 125 seconds including cleanup.

This is not a full-repository green-suite claim. Missing private July replay
fixtures, an absent external c603 checkout, superseded authority/adoption
ceremony tests, and extended research tests are explicitly excluded. Their
original failures and incomplete batches remain recorded. No retired approval
tokens or private evidence were manufactured. Library warnings and the test
selection's exact exclusions remain in XML/logs and the final inventory.

## Official guide and source audit

The single official book is built from `docs/main.tex`; its tuning chapter is
`docs/chapters/ch21b_hmc_tuning_interfaces.tex`. The Markdown
`docs/reference/hmc-tuning-interface.md` remains the agent/API entry reference
to the same procedure. The validation README and coverage table are synchronized.

The book now distinguishes the independent-look Gandy--Scott validation wrapper
from operational tuning and posterior stopping; reports the M15--M17 model,
fit, power and failure denominators; and keeps source versions and unavailable
consumer evidence explicit. The table is limited to suite reports. Its original
unavailable external designs do not negate later separately bound posteriordb
comparisons. Fixed nonlinear maps are never described as trained transports.

Technical sources and available author code were inspected for Gorinova,
Pakman--Paninski, and Afshar--Domke. Their previously unresolved citations now
resolve. The adjacent constraints discussion distinguishes a continuous gradient
kink from a potential jump and states that learned parameterization is frozen
before HMC. The source note records the inspected sections, code commits,
unavailable Afshar implementation, and a rejected wrong-paper retrieval.

The final PDF has 570 pages, SHA-256
`a6eec0ec1e578c84765bf735a8806af496e97dad492f4c53ed5c56a3a1475a8d`.
The prior 567-page PDF and all three versioned builds are preserved. Changed
discussion and table pages were rendered and inspected; a confirmed overflowing
chapter header was repaired. All 16 existing displayed equation blocks in the
two touched chapters remain verbatim relative to the Git baseline. There are
no unresolved citations or references. Existing typography warnings elsewhere
remain in the log; this is not a whole-book typography or human-voice certification.

## Decisions and remaining questions

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep scoped checkpoint specialization | Exact same-source numerical and hash parity | No mismatch or corruption-check regression | Two supplied-geometry fixtures; descriptive host cost | Preserve regression tests; profile future actual bottlenecks | General sampler speedup |
| Keep lazy export repair and migrated public oracles | Focused tests and named regression inventory pass | No unresolved failure in selected tests | Explicit excluded historical fixtures | Maintain the current public-path test suite | Full-repository green status |
| Install official guide | Source checks, resolved references, rendered inspection and source/PDF match | No unresolved changed-source citation | Scope limited to reviewed discussion/pages | Use the official book and matching API reference | Whole-book scientific certification |
| Close M13--M18 bounded execution | All planned phase work reconciled within budget | No active worker or invalid artifact | Difficult models and calibration remain unresolved | Follow the master remaining-gap agenda | All scientific gaps closed |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No invalid terminal M18 evidence; original test failures/timeouts remain preserved |
| Statistically supported ranking | None; exact intervention parity is the engineering criterion |
| Descriptive-only differences | Host medians, per-arm elapsed time and candidate counts |
| Default readiness | No new numerical, stopping, metric or learned-transport default is promoted |
| Next evidence needed | Target-specific geometry repair, delayed warmup/stopped-interval coverage, subtle whole-fit defect power, more precise sequential null-size estimates, and matched MacroFinance inputs |

M18 charges 4857.406613131985 CPU-reference worker seconds, including the
900-second inspection/tooling allowance and explicitly conservative timeout
charge, and 585.2183460368542 GPU worker seconds. These are within the phase
8000/2000 ceilings. Cumulative charges are 181663.99923514872 CPU and
155022.62885525584 GPU seconds. Remaining authorization is
77536.00076485128 CPU and 17777.37114474416 GPU worker seconds, approximately
21.54 and 4.94 hours. No prior phase charge is reset.

Terminal skeptical review: the strongest alternative explanation for favorable
results is the limited prepared-fixture scope and generous small-oracle moment
tolerances. These do not validate automatic preparation on arbitrary targets.
A changed numerical projection or lost live-corruption check would overturn
the hash specialization's admission; neither occurred. Missing exact consumer
inputs affect that comparison alone. Centered-funnel failures, mixture warmup
caps and retained precision caps reject the tested configurations, not the
research direction. Weak SBC sensitivity and finite sequential null precision
remain open scientific questions, not repaired engineering defects. No R-hat,
ESS or MCSE diagnostic changes tuning membership, ranking or recovery.
