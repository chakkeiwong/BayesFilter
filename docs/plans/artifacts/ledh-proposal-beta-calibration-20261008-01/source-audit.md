# Source audit for the beta-selection addition

Scope: the three-branch probability-selection mechanism, not an updated survey
of all adaptive importance samplers. No novelty, newest-method, best-method or
author-code replication claim is made.

Primary source: Hera Y. He and Art B. Owen (2014), Optimal mixture weights in
multiple importance sampling, arXiv:1411.3954v1, submitted 2014-11-14.
Local PDF: .localresources/papers/he-owen-optimal-mixture-weights-1411.3954.pdf.
The downloaded arXiv metadata and a complete text extraction are preserved.

| Manuscript statement | Inspected primary anchor | Classification |
|---|---|---|
| Convexity of mixture-weight variance objective | Theorem 3 and proof Section 7.5 | Source result specialized to no control variates; project beta is source alpha |
| Estimate variance objective using fixed pilot proposal | Section 4, equation (15), with zero control-variate coefficients | Source mechanism, specialized here |
| Restrict mixture probabilities away from unsafe faces | Section 4.1 | Source context; the exact predictive-ratio bound is derived locally |
| Scope-weighted normalization and fixed three-branch calibration | Not an author implementation claim | Local procedure; ratio estimate not asserted unbiased |
| Projected-gradient solver, simplex projection and gap certificate | Explicit derivation in the chapter | Local solver; the paper uses a different numerical optimizer |
| Score and full recursive filter performance | No supporting experiment | Unevaluated; local variance derivation does not imply it |

Read sections: technical setup and variance definitions, Theorem 3, Section 4
and equation (15), Section 4.1, Section 6 discussion, Section 7.5 proof and
references. No reliance on metadata/abstract alone. Author implementation code
was not inspected; source-code faithfulness is not claimed.

Backward candidates seen in the references: Hesterberg (1995) and Owen–Zhou
(2000) on defensive importance sampling; Veach–Guibas (1995) on multiple
importance sampling; Cappe et al. (2008) on adaptive mixtures; Boyd–Vandenberghe
(2004) on convex optimization. Their technical text was not inspected in this
bounded follow-up and no additional theorem is attributed to them.

Search/access record, 2026-10-08: two web-tool requests failed with upstream
HTTP 502. Direct curl fetched the primary arXiv metadata and PDF successfully.
A Semantic Scholar forward-citation metadata query returned HTTP 429; citation
counts and a forward ranking are unavailable. Forward coverage and exhaustive
erratum/publication-status checking are therefore incomplete. Those gaps limit
survey/newest/best claims; the fixed-source derivation is inspectable locally.
No numerical optimizer or empirical setting was adopted on citation counts.

Omission-risk review: other adaptation objectives and score-specific proposals
could select different probabilities. The chapter gives one justified local
likelihood-variance procedure and explicitly does not claim a uniquely optimal
calibration for the recursive filter or analytical score.
