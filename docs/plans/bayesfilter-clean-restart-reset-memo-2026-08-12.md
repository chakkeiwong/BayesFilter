# BayesFilter clean restart reset memo

## Date and scope

Date: 2026-08-12

This memo records the repository state after the generated-file boundary was
reviewed. It is the handoff point for a fresh BayesFilter agent. The task was
repository hygiene and restart preparation, not a scientific promotion or
default change.

## Skeptical audit

The initial worktree contained modified implementation and test code, authored
benchmark drivers and plans, durable evidence artifacts, downloaded literature,
and runtime payloads. A blanket stage would have mixed raw checkpoints, tensor
states, logs, and private execution state into the commit. Those files cannot
answer a promotion or claim question when the compact receipts and result notes
already preserve the relevant decision. The revised boundary therefore keeps
authored source, tests, plans, result notes, manifests, summaries, receipts,
hash ledgers, and other explicitly claim-bearing artifacts visible, while
ignoring reproducible runtime payloads and downloaded literature.

## Repository state

- Branch: `main`
- Remote: `origin` (`git@github.com:chakkeiwong/BayesFilter.git`)
- Before this cleanup, local `main` was nine commits behind `origin/main`.
- Existing tracked modifications were preserved; no unrelated changes were
  reverted.
- Authored untracked Python, Markdown, TeX, BibTeX, and test files are staged
  as source/evidence in the cleanup commit.
- The newly collected multimodal-HMC paper corpus remains local and ignored
  under `.localresources/papers/multimodal_hmc/`.

## Git boundary

Tracked files are the source of truth. New generated files are ignored when
they are execution state, including:

- Python caches, compiled objects, build directories, and test caches;
- LaTeX byproducts and rendered PDFs;
- raw sampler tensors, checkpoints, trainer states, progress files, logs,
  process IDs, JSONL streams, and private payloads;
- benchmark runtime directories and intermediate traces; and
- downloaded literature corpora that are not themselves claim artifacts.

Files that remain eligible for tracking are authored implementation and test
code, experiment plans, reset/result notes, survey sources, and compact
promotion/claim support such as `result.json`, `summary.json`, manifests,
receipts, decision records, selection/tuning records, and hash ledgers. Existing
tracked files stay tracked even if a later ignore rule would match them.

The semantic filename rules are a screening convention, not scientific
validation. A result or manifest is not promoted merely because Git tracks it;
the governing plan and result note must still establish its evidence contract,
veto status, uncertainty, and nonclaims.

## Restart instructions

1. Start from `main` after confirming the cleanup commit is present locally and
   on `origin/main`.
2. Read this memo and the most recent lane-specific reset memo before running
   any serious campaign.
3. Use a fresh, versioned output root under `docs/plans/artifacts/` for each
   run. Write compact terminal evidence outside ignored runtime subdirectories.
4. Keep the research intent ledger, evidence contract, attempt budget, stop
   conditions, run manifest, and result note together in `docs/plans/`.
5. Before a claim-bearing run, verify `git status --short` has no unclassified
   generated file. If a new output type appears, classify it and update
   `.gitignore` before continuing.

## Scientific nonclaims

This cleanup does not establish posterior correctness, HMC readiness,
statistical superiority, source faithfulness, production readiness, or any
change to BayesFilter defaults. Ignoring a payload does not discard its local
debugging value; the compact receipts and result notes must identify any
ignored payload by path or hash when it is needed to reproduce a decision.

