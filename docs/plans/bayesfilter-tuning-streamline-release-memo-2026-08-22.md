# BayesFilter Tuning Streamline: Release Memo

Date: 2026-08-22
Status: RELEASE CANDIDATE — pending owner sign-off on Section 6 items
Evidence chain: handoff (2026-08-19) → Steps A–C result (2026-08-20) →
Step D adjudication + §6a contract gap (2026-08-20/21) → Step D–H result
(2026-08-21) → this memo. Census artifacts:
`/tmp/bf-release-census/` (durable).

## 1. What This Release Delivers

- Exactly two active, artifact-authoritative HMC tuning interfaces:
  `bayesfilter.inference.tune_hmc_kernel` and
  `bayesfilter.inference.tune_fixed_transport_hmc_kernel`, enforced by the
  route registry and AST inventory guard (10 routes, 0 unclassified,
  0 stale).
- Public compatibility facades complete, including the new
  `build_bootstrap_fixed_mass_adapter` (identity-preserving alias).
- All expressible consumer migrations executed: 19 behavior-preserving
  private-import swaps (dsge_hmc ×4, MacroFinance ×15) and the MIDAS robust
  driver on the canonical interface with its new contract test.
- Frozen-lineage registry: 2026-08-14 MIDAS robust campaign, June-2026 C2
  phase4/5T and CCMA owned-client campaigns, 2026-08-08 dsge repair
  one-shots, r16 Rotemberg lineage — all preserved, none able to issue new
  canonical admission artifacts.
- Rebaselined `CURRENT_SOURCE_HASHES` freshness pin (provenance comment in
  source).

## 2. Verification Summary

| Gate | Result |
| --- | --- |
| Focused cross-repo gates (two green runs) | MacroFinance 74/74, dsge_hmc 50/50 (run 2, 2026-08-22, incl. Step E + pin tests) |
| BayesFilter GPU canary | 108/108 GPU-visible, memory growth verified pre-init on both RTX 4080 SUPER |
| Route inventory guard | clean |
| Full-suite failure census (tfgpu-full, CPU-hidden, per-file) | MacroFinance: 234 failures / 90 of 609 files. dsge_hmc: 412 failures / 113 of ~511 censused files. **Zero failures in any campaign-modified file or its tests, in either repository.** |

## 3. Known Issues Ledger (pre-existing, out of campaign scope)

MacroFinance (234 failure identities in `MF_failures.txt`):
- `test_three_currency_block_tfp_hmc.py` (30) — largest single cluster.
- Mixed-frequency TFP family (~50 across 8 files: generated-data adapter,
  phase4 SVD pilot/backends, phase5 pilot, synthetic recovery/contract).
- `test_run_ccma_broad_fixed_metric_l_epsilon_search.py` (8) — the
  documented 2026-08-16 "historical-v3/live-v5 acceptance-evidence"
  family, deliberately outside the handoff repair scope.
- Matched-DGP artifact-freshness family (`migration_adapter`,
  `posterior_runtime_validation`) — artifacts no longer rebuild against
  current code; reproduces in canonical `tfgpu`.
- Order-dependent collection/import errors (7 files) plus `sys.modules`
  pollution failures that pass in isolation.

dsge_hmc (412 failure identities in `DS_failures.txt`):
- Rotemberg policy-training contract families (~130: raw-policy structural
  parameterization 42, direct-policy worker 30, feasibility objectives 34,
  rp05 training 14...) — belongs to the unrelated dirty Rotemberg work.
- NK budget-capacity phase-package family (~40 across phases 6–10).
- EZ-NK learned-global controller redesign (16).
- Six execution-time segfault files (named in the Step D–H result §5b):
  BGS public-integration/synthetic-generator, d296 likelihood-gradient/
  state-space, and the SVD-SSM HMC-recovery pair. Release action: add to
  an explicit exclusion list mirroring the `tests/archive` precedent.
- Six long-runner files exceeding 15 min single-file (candidates for
  `slow`/`overnight` markers).

Interpretation discipline: these are engineering census results from dirty
worktrees in the unpromoted `tfgpu-full` clone. They establish which files
fail and where, not why; no scientific claim is affected.

## 4. Explicitly Not In This Release

- C2 phase4+5t and CCMA owned-client migrations (Step D §6a contract gap:
  caller-owned candidate grids under externally frozen mass are not
  expressible via the canonical interfaces; frozen campaigns preserved).
- Step H quarantine of historical routes from default exports (blocked by
  the same sanctioned consumers).
- Any repair of the Known Issues Ledger families.
- Any posterior-correctness, convergence, sampler-superiority, statistical
  ranking, or GPU-performance claim.

## 5. Proposed Commit Packaging (awaiting owner approval; nothing committed)

1. BayesFilter: `feat(inference): add public build_bootstrap_fixed_mass_adapter facade`
   — `hmc_bootstrap.py`, `__init__.py`.
2. BayesFilter: `docs(plans): tuning-streamline step A–H results, adjudication, release memo`
   — the five campaign documents.
3. dsge_hmc: `refactor: swap private mass-signature imports to public facade`
   — 4 files.
4. dsge_hmc: `fix(tests): align relaunch contract with committed selection rule; add frozen-transport score methods`
   — the 2 patch-bundle files (from Step B).
5. MacroFinance: `fix(tests): audited tuning-contract drift repairs`
   — the 2 patch-bundle files (from Step B).
6. MacroFinance: `refactor: private tuning imports to public facades`
   — 15 files.
7. MacroFinance: `feat: MIDAS driver on canonical tune_hmc_kernel + contract test`
   — driver + new test.
8. MacroFinance: `chore: rebaseline recovery-phase1 source-hash pins`
   — 1 file.

Unrelated dirty files in all three repos stay uncommitted and untouched.

## 6. Remaining Owner Decisions Before Tagging a Release

1. Approve the commit packaging above (and decide branch/PR flow).
2. §6a contract-gap direction (fixed-transport reformulation vs. canonical
   collapse for the next C2/CCMA campaign) — or explicitly defer with the
   frozen-campaign classification standing.
3. Environment policy: promote `tfgpu-full` (pandas 3.0.5, torch
   2.13.0+cpu, gymnasium 1.3.0) as the pinned test environment, or keep
   `tfgpu` canonical with the census recorded as clone-only evidence.
4. dsge_hmc segfault exclusion list + slow-marker patch: authorize me to
   apply it, or hand to the Rotemberg/BGS owner (files belong to that
   workstream).
5. Disposition of the Known Issues Ledger: ship as release notes (my
   recommendation) or gate the release on repairing selected families.
