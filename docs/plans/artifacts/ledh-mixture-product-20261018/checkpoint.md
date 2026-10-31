# Product moment repair checkpoint — 18 October 2026

- Branch `ledh-refactor-with-policy-fix`, checkout `ledh-iapf-execution-20260922`,
  base `ea1018de1da55ed02a29a8d60f08a6827fce8a42` plus preserved dirty work.
- Stage COMPLETE under `docs/plans/ledh-mixture-product-moment-repair-20261018.md`.
  Question: does exact conditioning of the approximate observation Gaussian
  by a frozen mixture improve the global consumer's model-score accuracy?
- Shared multidimensional batched moments and analytical derivatives implemented
  as optional `mixture_product_moments`. Old mode remains default. Twenty focused
  checks pass, then two extended map checks pass; total test time 34.49 seconds.
- Fresh GPU/XLA run completes 24 fits and 2,304 rows. Terminal audit passes
  13,824 trace checks, 74 source hashes and 24 finite-difference checks.
  Largest FD discrepancy 7.5112631e-7 passes the mixed tolerance.
- Four simultaneous intervals support product/global improvement over the old
  fitted/global consumer; none supports improvement over unguided global.
  Product/global and product/equal have seven and 21 descriptive heuristic
  losses. Both numerically viable, neither promoted. No continuation veto.
- MathDev verifies scoped scalar identities; all 19 matrix-proof obligations
  remain parser-inconclusive. Aggregate lexical mismatch is not a refuted
  equation. Detailed classification and scientific limits: `result.md`.
- Evidence: `attempt-01/{manifest.json,result.json,rows.jsonl,analysis.json}`;
  focused logs and MathDev responses in this directory. Pre-run plan unchanged.
- Budget CLOSED: 407.033164738/900 GPU-process seconds, 1/2 launches;
  1.562513115/180 CPU diagnostic seconds; 34.49/300 focused-test seconds.
  Document build/review accounting is in `document-build/review.md`.
  No live process or transfer of unused budget.
- Exact next action: plan a fresh paired comparison of exact seven-component
  and fitted future messages under the SAME product-moment consumer, retaining
  unguided global and old-consumer controls. Distinguish conditional saved-fit
  mechanism checks from fresh-fit confirmation. Do not enlarge capacity first.
- OT1/5 remains an untuned comparator. No default/canonical/HMC/original-KDM,
  commit, push or cross-branch action follows.
