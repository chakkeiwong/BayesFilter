# Public MacroFinance GPU qualification: terminal result, 29 October 2026

The nine-candidate campaign completed successfully at **11:51:16 Asia/Shanghai
on 29 October 2026**, after **38325.626110966085 seconds (10h 38m 46s)** of
inclusive supervisor time. Both recorded processes have exited. The current
qualification is complete; the overall modern-HMC master remains incomplete.

## Evidence and audit

The [reviewed plan](bayesfilter-modern-hmc-public-qualification-2026-10-29.md)
asked whether the source-bound public GPU target supports the complete frozen
cohort, native trial accounting, fresh verification, export and reload.
The terminal evidence answers yes for this target and configuration.
The comparator was C1's declared execution invariants, not its numerical
outcomes. No numerical policy, reference, candidate set or threshold changed.

Campaign root: artifacts/modern-hmc-implementation-2026-10-07/N5/public-qualification-20261029-01.
Its launch.json, qualification-01/manifest.json and evidence/manifest.json
record exact commands, clean commit f9291ee4437aa8c00dabbe331bffc6871b574c5c,
frozen consumer/input identities, seeds, environment and GPU policy. The run
uses TensorFlow/TFP, GPU/XLA float64, TF32 disabled, verified memory growth and
fresh streams. R-hat stayed reporting-only in tuning; NUTS was excluded.

The skeptical terminal audit reconciled 1,248 independent returned trials
across 48 completed work items. All 1,248 native trial identifiers and seeds
are unique; no attempted return is missing and no partial chunk remains.
Charged and expected gradient work agree at 9,977,856. All 629 frozen source
hashes still match. The controller, audit, exported membership and reload
result contain exactly the same seven verified candidates. All seven reloads
preserve endpoints and produce finite short continuation. Short continuation
checks mechanics; it does not establish posterior convergence.

| L | Epsilon | Measurement trials | Fresh verification trials | Outcome |
|---:|---:|---:|---:|---|
| 3 | 0.4 | 64 | 64 | Verified, exported and reloaded |
| 3 | 0.8 | 16 | 0 | Rejected: movement screen |
| 5 | 0.4 | 64 | 64 | Verified, exported and reloaded |
| 5 | 0.8 | 16 | 0 | Rejected: movement screen |
| 9 | 0.4 | 64 | 64 | Verified, exported and reloaded |
| 13 | 0.4 | 128 | 128 | Verified, exported and reloaded |
| 18 | 0.4 | 128 | 128 | Verified, exported and reloaded |
| 25 | 0.2 | 32 | 32 | Verified, exported and reloaded |
| 25 | 0.4 | 128 | 128 | Verified, exported and reloaded |

The two .8 candidates failed movement checks: 15 of 16 measurement trials for
L=3 and 16 of 16 for L=5. Their evidence was valid, with no engineering
invalidity or hard-veto record; these are candidate-local promotion failures.
They did not stop evaluation or erase other candidates. All seven verified
candidates passed the declared acceptance/health screens; none is ranked best.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Close scoped public GPU qualification | Complete native cohort, accounting, export and reload pass | No campaign-invalidity finding; two movement rejections preserved | Limited target, source, geometry and starts | Preserve results; advance to C3 design | Universal tuner robustness |
| Retain all seven verified candidates | Fresh replicated qualification and health screens pass | No recorded promotion veto on these members | Posterior exploration and precision untested here | Assess under an explicit posterior/reference contract | Convergence, ranking or release readiness |
| Reject the two .8 candidates in this scope | Movement requirements fail | Candidate-local veto, not direction rejection | Other epsilon/geometry choices not tested by this frozen design | Preserve outcome; do not rerun merely to erase failure | Randomized HMC is invalid |

| Inference status | Result |
|---|---|
| Hard veto screen | No engineering-invalidity or hard-veto observations. Two declared movement promotion vetoes remain. |
| Statistically supported ranking | None; passing qualification is not a superiority comparison. |
| Descriptive-only differences | Runtime, candidate acceptance and differing replication counts do not rank methods. |
| Default-readiness | Not established. Modern-profile claim eligibility remains restricted. |
| Next evidence needed | C3 independent-reference posterior/precision contract and retained assessment; C4 estimator/tail reliability; C5 independent SIR; C6/N8 reconciliation. |

The strongest alternative explanation is that success is limited to this
frozen geometry and start bank. New posterior disagreement, an invalid source
identity or inconsistent reload would overturn stronger interpretations. The
weakest remaining evidence is posterior/reference coverage. This result does
not invalidate the target, implementation or research direction; it closes
one engineering qualification and identifies two rejected candidates.

## Settlement and next phase

The supervisor charged 38325.626110966085 seconds once, releasing
56106.373889033915 seconds of the 94432-second reservation. The remaining
unreserved upper bound is **186372.1157282945 seconds (51.77h)** after keeping
the 3600-second engineering and 7200-second preparation withholdings. This is
unused existing funding, not new credit, and remains an upper bound pending
final overhead settlement. Child durations must not be charged again.
Separate SIR funding stays 7363.064379416 seconds.

Next work is the C3 posterior/reference design: explicit estimands, precision
requirements, uncertainty-qualified independent reference, fresh streams and
shared sequential-controller use. Historical truth-recovery checks cannot
substitute for that reference. C2 wider migration/P19 dependency, C4 estimator
reliability, C5 independent fits and C6/N8 integration/book/release remain open.
No additional campaign was launched during this terminal review.

The [compact evidence index](../validation/modern-hmc-20261029-public-gpu/README.md)
now includes terminal-review.json and terminal-inventory.json. Prior launch
records remain immutable; the compact terminal packet does not duplicate the
large raw numerical evidence.
