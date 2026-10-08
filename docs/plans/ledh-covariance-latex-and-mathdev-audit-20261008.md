# Self-contained covariance proposal and MathDevMCP audit

Date: 2026-10-08. Branch: sqmc-development; base 209223fdd.
User request: fully document the proposed general filter in LaTeX and audit it
with MathDevMCP. This is document and mathematics work, not authorization to
promote a new runtime or launch a filtering campaign.

The reader knows Kalman/UKF filtering and importance sampling, but should not
need the preceding conversation. The note must explain which covariance is
conditional, predictive, UKF-updated, or importance-weighted; show exactly where
each enters; derive the proposal densities, moment reset, bounded higher-moment
repair and total analytical score; and provide executable-order pseudocode.
The earlier Markdown proposal and its source ledger are preserved as the
baseline. The LaTeX will identify corrections or added assumptions explicitly.

Narrative: the AR(1) variance puzzle leads to total covariance, then to a common
UKF proposal and local conditional flows. Importance correction determines the
weighted moments. Resampling preserves them; bounded rotations adjust higher
moments. Derivatives follow the same finite computation. Literature and limits
are explained where they affect these choices.

Deliverables: a standalone source and compiled PDF under docs/papers; a concise
audit result note and MathDevMCP raw reports under a versioned docs/plans/artifacts
directory; an updated recovery checkpoint. Human readability review remains
pending; this does not block the requested complete draft.

Skeptical audit before drafting: a correct covariance identity does not prove
likelihood/score accuracy. A map fitted to its own sampled input needs its full
Jacobian; the density derivation therefore fixes maps before sampling. A
parameter-dependent branch sampler would break the stated pathwise derivative;
fractions and labels are frozen. UKF normalization must not be multiplied into
the likelihood again. Exact second moments do not guarantee valid support or
correct higher moments. Orthogonal repairs have feasibility/orientation limits.
These conditions will be explicit, not hidden inside pseudocode. No baseline
or scientific ranking will be changed.

Audit contract: inspect every mathematical section manually, run MathDevMCP
document rigor/derivation tools on the final source, resolve substantive findings,
and use bounded symbolic checks for algebra that the LaTeX parser cannot certify.
Preserve parser abstentions separately from mathematical counterexamples. A
successful tool call or compilation is not a proof certificate for the filter.
Record exact coverage, source hash and limitations rather than calling an
unsupported automated result a pass. Check the rendered PDF and all references.

Budget: one initial document audit, one focused revision audit if needed, up to
12 bounded algebra checks; no GPU or scientific experiment. Local build retries
and syntax fixes remain within this documentation task.

Next action: deliver the completed PDF, LaTeX, and audit record.


Completed 2026-10-08: the standalone 19-page PDF and LaTeX are in docs/papers.
The initial 49 labelled equations were processed in 13 disjoint MathDevMCP
batches because both bulk report formats rejected more than four labels.
A focused four-label revision audit and 11 bounded symbolic attempts followed;
nine symbolic equalities were certified and two derivative syntaxes were
not encodable. Automatic inverse/Neumann flags are explicitly dispositioned,
not presented as a full-document proof. Rendering and reference checks pass.
No runtime, filtering experiment, model default or baseline was changed.
Full results: docs/plans/ledh-covariance-latex-mathdev-audit-results-20261008.md.
