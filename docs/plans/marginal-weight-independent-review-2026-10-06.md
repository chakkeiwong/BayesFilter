# Independent review of the marginal-mixture LEDH extension

User directive: review the supplied derivation, execution plan and results;
advise whether the other agent should merge. The user also authorized 48 more
hours for the continuing reference work. This review does not itself merge,
change defaults or execute another serious campaign.

Reviewed checkout: `/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917`,
commit `f5e69d716` (optional marginal weights). Review artifacts:
`docs/plans/artifacts/marginal-weight-independent-review-20261006-01/`.

Question: is the one-child-per-ancestor mixture importance formula valid,
does its analytical score differentiate the actual finite value computation,
and is an optional merge justified separately from default promotion?

Audit before execution: the comparator for engineering checks is identical
inputs through the same optional policy, changing batch width only. No comparison
to an oracle or inference about filter accuracy follows from this check. The
existing five focused mixture tests compare density/tangents to independent
TFP and finite differences and verify the actual flow map. Add a bounded CPU
FP64 XLA check of two distinct M13 parameter rows, versus separate single-row
calls and a repeated batch call. N=16, T=2 is a mechanics fixture, not a tuned
scientific setting. Stop on invalid status or numerical mismatch. No attempt
to tune controls to pass. At most 300 seconds total for these local checks;
CPU is intentional and GPU visibility is hidden before TensorFlow import.

Skeptical audit passes for that narrow question. Wrong-baseline and proxy risks
are controlled by retaining the identical model, data, random design, settings,
dtype and policy. A pass supports only call-chain parity at this fixture, not
GPU reproducibility, general batching, HMC or reference accuracy. Existing
controls are a convenient mechanics fixture, not reviewed tuning defaults;
validity is the early diagnostic. No stochastic ranking is attempted.

The mathematical review separately checks the outer ancestor weight, actual
affine Gaussian pushforwards, support/invertibility, all total-derivative terms,
and the limitation of the random-label Rao–Blackwell variance argument for
one-per-ancestor sampling. The source is Klaas/de Freitas/Doucet (2005), Section
3, Figure 3 and Section 3.1, plus the supplied derivation. Full-text paper is
already cached in the other checkout. This is a bounded method review, not a
literature completeness or current citation-metadata audit.

The additional 48-hour authorization is recorded separately from the completed
47.004-hour campaign. No previous failed run or the other agent's experiments
are silently charged to or erased from that ledger. The next serious reference
phase needs its concise updated evidence contract before launch.
