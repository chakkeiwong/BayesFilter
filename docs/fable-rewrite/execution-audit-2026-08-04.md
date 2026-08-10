# Fable rewrite execution audit

- **Date:** 2026-08-04
- **Scope chosen:** current 55-chapter source tree (`docs/main.tex`) only.
- **Deliverable:** a standalone rewritten monograph source + built PDF under `docs/fable-rewrite/`.
- **Why this scope:** the owner explicitly requested the rewritten document, not another planning artifact. Current-source-only defects (e.g. `ch26c`, expanded `ch32c`, expanded `ch32c2`) are in scope; historical 54-chapter PDF reconstruction is not.

## Skeptical audit before execution

### Risks in executing the full master plan literally
1. The full plan is too large for one safe pass and includes source-audit and literature-snowball tasks whose outputs are new research artifacts, not immediate document text.
2. Some repairs (e.g. the third squared-TT/Jacobian issue, full signed downdate derivative exposition) remain under-specified and should not be silently "fixed" without a separate derivation pass.
3. Structural archival moves (`chapters_restart_staging/`) would be unsafe because alternate tracked roots exist.

### Bounded execution choice
This execution will therefore:
- **not** touch the original `docs/` monograph sources;
- build a **standalone rewrite tree** under `docs/fable-rewrite/`;
- apply the highest-confidence, directly actionable fixes that were confirmed by the review cycle and do not require new ambiguous mathematical design decisions;
- preserve unresolved under-specified items as explicit nonclaims or clarified wording rather than inventing unreviewed fixes;
- prioritize a **successful build and readable PDF** over exhaustive closure of every lower-priority review note.

### What is included
- build repair (missing figures issue via tracked PNG route if needed);
- citation/bibliography repair where the needed metadata is already known from the review record;
- major wording/logic fixes in `ch03`, `ch13`, `ch17`, `ch19`, `ch19b`, `ch20`, `ch32c`, `ch32c2`, `ch32e`, `ch34`, `ch35`, `ch35b`, `ch38`, and the relevant appendices;
- label cleanup where straightforward and low-risk.

### What is excluded from this execution pass
- full historical 54-chapter PDF reconstruction;
- moving or archiving alternate-root source trees;
- full literature snowballing and omission-ledger completion;
- new theorem-level derivations not already settled by the review (e.g. a complete signed rank-one downdate derivative calculus);
- speculative fixes where the review ended at "under-specified" rather than "confirmed defect".

### Success criteria for this bounded execution
1. `docs/fable-rewrite/main.pdf` builds successfully.
2. The rewritten source reflects the core confirmed repairs above.
3. The rewritten text is more honest about targets, source support, and artifact boundaries than the original.
4. The original `docs/` tree remains untouched.
