# Phase 38 Repair and Refresh Note

| Attempt | Failure class | Repair | Result |
|---|---|---|---|
| initial checkpoint proposal | design defect: terminal-only receipts could not evaluate selected checkpoints on audit rows | extend the runner to emit audit moments at predeclared checkpoints; add a read-only reporter | repaired before claim-bearing interpretation |
| identity trace | none | fresh N=256 M0 root, steps `{1,5,10,25,50,100,150,200}`, unchanged theta measure | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` |
| affine trace | none | same frozen protocol with train-measure affine map | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` |
| checkpoint report | none | validation-only score; audit evaluated post-selection | `PASS_CHECKPOINT_SELECTION_AUDIT_RECEIPT` |

The affine wide arm selected step 150 and had descriptively smaller audit
moment residuals than its step-200 terminal state. The other arms selected
step 200. Residuals remain material, so this phase does not promote whitening
or a NeuTra route.

The design and runner repairs were local and did not change target, measure,
proposal, hardware class, or campaign gates. No continuation veto fired. The
next repair is a read-only train/validation/audit empirical-measure separation
report before any objective or data-generation change.
