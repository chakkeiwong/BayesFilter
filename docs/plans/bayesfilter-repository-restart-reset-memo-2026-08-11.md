# BayesFilter Repository Restart Reset Memo

Date: 2026-08-11
Status: `CLEAN_RESTART_SNAPSHOT`

## Restart Point

The repository hygiene pass classifies every present path as tracked or
ignored. Authored source, tests, plans, derivations, literature anchors, the
live monograph rewrite, and compact artifacts that support a retained claim,
veto, tuning decision, or promotion decision are tracked. Reproducible build
products, copied source snapshots, mutable runtime state, and routine,
intermediate, failed, or superseded run outputs are ignored.

The push commit containing this memo is the restart authority. After restart,
verify the exact commit with:

```bash
git status --short --branch
git log -1 --oneline --decorate
```

Expected state: branch `main`, synchronized with `origin/main`, with no
untracked or modified non-ignored files.

## Active Scientific State

Three workstreams are preserved in the snapshot:

1. GenUT higher-moment filtering and NeuTra integration, including batched
   TensorFlow targets, scope-specific tuning infrastructure, and the retained
   four-model and Austria diagnostics.
2. Zhao-Cui fixed-variant value/score work, including Austria SIR bounded
   teachers and rank-one proposals plus actual-SV analytic/manual score
   derivations and tests.
3. The review-driven monograph rewrite and its audit/certificate ledgers. The
   live rewrite is under `docs/fable-rewrite/monograph/`; the generated
   pre-rewrite copy is ignored because Git history preserves the prior source.

## Scientific Verdicts And Nonclaims

- Repository hygiene does not promote any numerical method or establish a new
  default.
- The retained GenUT comparisons contain viable and rejected candidates, but
  no statistically supported universal ranking. Descriptive stochastic
  differences remain descriptive.
- Austria SIR caps and proposal repairs remain scope-specific
  `extension_or_invention` work unless a cited Zhao-Cui paper and author-source
  route establishes otherwise.
- Actual-SV analytic/manual score work remains governed by its derivation lock,
  finite-difference/parity checks, and result note. This snapshot alone does
  not establish HMC, default, production, or scientific readiness.
- Contract E--Chol remains the only canonical LEDH reset route; no historical
  raw-barycentric artifact is upgraded by being retained.
- Every future claim-bearing LEDH run still requires its own matching offline
  tuning artifact and the canonical transport chunk policy.

## Artifact Policy

Track generated artifacts only when an authored result, decision, reset, or
promotion note identifies them as evidence needed to reproduce its claim or
veto. Ignore generated artifacts when they are build products, local caches,
copied snapshots, raw sampler state, private/mutable progress, or routine,
failed, superseded, exploratory, smoke, capacity, training, replay, or parity
attempts not cited by retained claim-bearing documentation.

Narrow `.gitignore` exceptions retain a smoke result only when a current
authored note cites that exact artifact as explanatory or veto evidence. A
semantic filename such as `result.json` or `run_manifest.json` is not by itself
sufficient for future tracking.

Downloaded papers under `.localresources/papers/` are tracked when they
materially anchor the implementation or scientific decision, consistent with
the repository literature policy. Generated PDFs under `docs/` remain ignored;
the removed `docs/main.pdf` must be rebuilt locally from tracked TeX sources.

## Verification Ledger

| Check | Status |
|---|---|
| `git diff --cached --check` | pass |
| Python compile check for `bayesfilter`, `tests`, benchmark harnesses, scripts, and the squared-TT certificate checker | pass |
| Initial focused CPU-only suite | `146 passed, 1 failed`; the failure exposed stale `SVX-ZC` executable-registry classification |
| Registry repair | `SVX-ZC` moved to blocked until explicit XLA/HMC admission and fresh scope-specific tuning |
| Directly affected CPU-only rerun | `67 passed` |
| Custom TensorFlow op | ignored `.so` rebuilt against the installed TensorFlow after an ABI-stale import failure; import then passed |
| Staged secret-pattern scan | no finding |
| Staged object-size scan | no object over 10 MB |
| Literature validation | staged `.pdf` files identified as PDFs; one HTML response mislabeled as the Huber PDF is ignored |
| `git ls-files --others --exclude-standard` | empty |

All test commands deliberately set `CUDA_VISIBLE_DEVICES=-1`; they are CPU
correctness checks and are not GPU evidence. The final commit hash, upstream
reconciliation, and post-push branch status are supplied by Git after this memo
is committed; this file does not self-attest a hash that cannot exist yet.

## First Restart Action

Read this memo, then inspect the newest plan/result for the workstream being
continued. Before any new non-trivial experiment, perform the required
skeptical plan/default audit and confirm the evidence contract, scope-specific
tuning identity, compute budget, versioned output root, and continuation vetoes.
Do not infer a scientific continuation solely from the fact that the repository
is clean.
