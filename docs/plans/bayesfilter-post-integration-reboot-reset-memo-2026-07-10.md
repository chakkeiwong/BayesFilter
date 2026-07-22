# Reset Memo: BayesFilter Post-Integration Reboot

## Date
2026-07-10

## Context
This memo prepares a clean reboot after the July 2026 LGSSM/Neutra HMC and
LEDH score/evidence integration pass.

The pre-memo integration baseline is:

- Branch: `main`
- Remote: `origin/main`
- Commit: `d269f5bbd8531b878d4f25897a357fbc8f172488`
- Subject: `Integrate LGSSM HMC and LEDH evidence artifacts`

That commit was rebased onto the then-current `origin/main` and pushed
successfully. The rebase had one conflict in
`tests/test_hmc_kernel_tuning_public_api.py`; the resolution kept both sides:
the remote `handoff_screen_policy` assertions and the local verification
chunk/min-retained assertions.

## Decision / policy
Future sessions should assume the repository hygiene policy is:

- Generated files that do not support a claim, gate, promotion, handoff, or
  reproducibility record should be ignored.
- Evidence artifacts that support a claim/promotion and are no larger than
  20 MB should stay tracked.
- All visible files should be either tracked or ignored before reboot, commit,
  or handoff.
- Private generated benchmark arrays under benchmark `private_diagnostics`
  should remain ignored unless a reviewed plan explicitly promotes a bounded
  file as evidence.
- Public JSON, JSONL, Markdown, plan, review, and result artifacts under
  `docs/benchmarks`, `docs/plans`, and `docs/reviews` may be tracked when they
  support the evidence ledger.

Execution policy remains:

- GPU/CUDA/NVIDIA probes, GPU XLA tests, and GPU training runs require
  escalated/trusted execution.
- Deliberate CPU-only tests should hide GPU devices before framework import,
  for example with `CUDA_VISIBLE_DEVICES=-1`, and should record that choice.
- BayesFilter-owned differentiable algorithmic implementation defaults to
  TensorFlow/TensorFlow Probability, not NumPy.
- Neutra training policy is GPU-first; sample generation policy is CPU
  multicore unless a reviewed plan changes the boundary.
- XLA validation work should test `jit_compile=True` paths; do not treat
  `jit_compile=False` runs as production evidence for the XLA target.
- `tf.GradientTape` should not be used in production differentiable paths
  except as a diagnostic path recorded as such.

## What changed in the integration commit
High-level contents of `d269f5b`:

- Added and updated LGSSM/Neutra HMC infrastructure:
  - multidimensional triangular LGSSM testing support;
  - fixed-transport and affine-payload Neutra mechanics;
  - CPU multicore HMC chain/sample boundary harnesses;
  - deterministic LGSSM HMC tuning driver, configs, and result artifacts;
  - QR/SVD Kalman score wiring repair artifacts and result notes.
- Added and updated HMC tuning machinery:
  - budget ladder and kernel tuning public API extensions;
  - XLA/verification chunk controls;
  - handoff nomination diagnostics;
  - public/private tuning artifact separation.
- Added LEDH score-contract and evidence tooling:
  - `bayesfilter/highdim/ledh_score_contract.py`;
  - `bayesfilter/highdim/ledh_score_artifact.py`;
  - LGSSM, fixed-SIR, predator-prey, actual-SV, generalized-SV, and KSC-SV
    score/value benchmark scripts and tests;
  - compact-score/default and score-wiring repair runbooks, ledgers, result
    notes, and review bundles.
- Updated `.gitignore` so generated private benchmark arrays are ignored while
  bounded public evidence artifacts remain trackable.

The integration commit touched 383 files. Treat the relevant runbook/result
files as the authority for scientific or promotion claims; this memo is only a
navigation and reboot note.

## Verification already run
Before this reset memo was written, the following hygiene checks had been run
after pushing `d269f5b`:

```bash
git status -sb
```

Observed:

- `## main...origin/main`

```bash
git rev-parse HEAD
git rev-parse origin/main
```

Observed:

- Both resolved to `d269f5bbd8531b878d4f25897a357fbc8f172488`.

```bash
git ls-files --others --exclude-standard
```

Observed:

- No output. There were no untracked, non-ignored files.

```bash
git diff --check
```

Observed:

- Passed with no output.

```bash
python - <<'PY'
import pathlib
import subprocess

files = subprocess.check_output(
    ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD'],
    text=True,
).splitlines()
large = []
for f in files:
    p = pathlib.Path(f)
    if p.exists() and p.is_file():
        size = p.stat().st_size
        if size > 20 * 1024 * 1024:
            large.append((size, f))
print('large_count', len(large))
PY
```

Observed:

- `large_count 0`

```bash
git push origin main
```

Observed:

- `52ee244..d269f5b  main -> main`

## Known limitations / cautions
- A focused collection check for
  `tests/test_hmc_kernel_tuning_public_api.py` did not run tests. It failed at
  import time with
  `ModuleNotFoundError: No module named 'tests.test_hmc_kernel_tuning_fixed_mass_step'`,
  including after retrying with `PYTHONPATH=.` and `CUDA_VISIBLE_DEVICES=-1`.
  Do not interpret that as an HMC runtime failure or as a passing test. It is
  an unresolved test import/collection issue.
- No full test suite was run as part of the final git hygiene push.
- No new GPU training, HMC sampling, leaderboard rebuild, or scientific
  promotion run was performed during the final hygiene/push step.
- The committed runbooks and result notes contain many CPU-hidden wiring checks
  and bounded GPU/XLA gates. Re-check the exact artifact before making any
  claim about posterior correctness, HMC readiness, default readiness,
  source-faithfulness, or scientific superiority.
- Claude review calls have sometimes been blocked by external-disclosure
  policy. Where result files record Codex substitute reviews, treat them as
  substitute review evidence only.

## Suggested next steps

### Operative HMC Terminal Override, 2026-07-13

This section supersedes the entire 2026-07-12 HMC override below for current
execution decisions.

- Read
  `docs/plans/bayesfilter-hmc-semantic-identity-migration-visible-stop-handoff-2026-07-11.md`
  first, then its master program, runbook, ledger, Phase 7 academic result, and
  Phase 8 closeout result.
- The typed-identity academic Phase 7 campaign executed once with two CPU/XLA
  workers, four chains, and the fixed transition/execution identities.
- Attempt 1 reached the declared `16000` burn-in cap. Diagnostics were finite,
  bulk and tail ESS passed, but eight of 18 parameters failed R-hat `<=1.01`;
  maximum R-hat was `1.043456525609825`.
- Retained sampling did not begin. No retained samples, posterior-recovery
  result, Phase 8 scientific runtime, or NeuTra execution exists.
- Terminal classification is `diagnostic_cap_failure`, exit code `1`. The
  result and checksum manifest verify, no Phase 7 process remains, and retry is
  forbidden by the active campaign state machine.
- Phase 8 completed documentation and boundary closeout only. The semantic-
  identity migration program is closed.
- Any further HMC work requires a new research/repair plan and user direction.
  The new plan must distinguish initialization, transition tuning, target
  geometry, and diagnostic-window explanations without weakening the terminal
  Phase 7 result.
- NeuTra remains a separate GPU-training lane. This HMC result neither
  authorizes nor scientifically rejects NeuTra.

Terminal records:

- `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-result-2026-07-13.md`;
- `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase8-closeout-result-2026-07-11.md`; and
- terminal result embedded hash
  `sha256:0724851756606956d2bf9d79fa62597fcef22a0c3c0737548d3383650306e076`.

### Operative HMC Override, 2026-07-12

The LGSSM HMC state below supersedes the older Phase 6AA resume pointers for
current execution decisions:

- Read
  `docs/plans/bayesfilter-hmc-semantic-identity-migration-visible-stop-handoff-2026-07-11.md`
  first, then its runbook, ledger, Phase 6 result, and reviewed subplan.
- Phase 6 attempt 1 permanently consumed its V2 smoke authority and claim, then
  failed before worker initialization with `runtime_error:BrokenProcessPool`.
  It produced zero worker PIDs, no HMC transition or diagnostics, and no private
  sample bytes. Preserve its complete 13-file evidence set exactly.
- The implementation failure was repaired and the V3 proposal pair passed
  frozen implementation and exact-artifact reviews. The exact V3-bound
  approval was received and consumed by one attempt-2 launch.
- Current gate: `AWAITING_HUMAN_PHASE7_SERIOUS_APPROVAL`.
- Attempt 2 passed
  `PASS_PHASE7_TYPED_IDENTITY_SMOKE_MECHANICS_ONLY_STOP_BEFORE_SERIOUS_APPROVAL`.
  Its result is
  `sha256:e7584e3c3d62e0a2370a33c1a77c8b9c6b1e157d1199cea4ceb9fd749a7a576d`
  and terminal output manifest is
  `sha256:805312c66c742cf2f7bce6da9c8e585a2bc99350ebd3bd65f474fd063eba51a8`.
- Two persistent Host-XLA workers completed four chains with 4 burn-in and 8
  retained transitions per chain. Protected sample shape, finiteness,
  provenance, hash, artifact cross-links, both attempts' evidence integrity,
  and process teardown verified.
- Smoke R-hat/ESS values are explanatory only. No convergence, recovery,
  ranking, default, GPU, or serious-readiness claim follows.
- Resume at
  `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-subplan-2026-07-11.md`.
  The active V2 config remains `runtime_authority=false`. The separate one-use
  serious implementation and proposal reviews pass. Stop for exact human
  approval bound to proposal manifest
  `sha256:c1f5709ee64eb898aa74e457553248ef32aac9bdc8a100b3f05f1431eebfa330`.
- The configured serious result path still holds the historical pre-migration
  blocker. Its exact immutable archive and terminal archive manifest now pass;
  controlled replacement remains prohibited before durable claim consumption.
- Serious Phase 7, Phase 8, and NeuTra remain unauthorized.

1. For a fresh reboot, start with:

   ```bash
   git fetch origin main
   git status -sb
   git rev-parse HEAD
   git rev-parse origin/main
   git ls-files --others --exclude-standard
   ```

2. If resuming LGSSM/Neutra HMC, read these first:

   - `docs/plans/bayesfilter-hmc-semantic-identity-migration-visible-stop-handoff-2026-07-11.md`
   - `docs/plans/bayesfilter-hmc-semantic-identity-migration-visible-runbook-2026-07-11.md`
   - `docs/plans/bayesfilter-hmc-semantic-identity-migration-visible-ledger-2026-07-11.md`
   - `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase6-smoke-result-2026-07-11.md`
   - `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md`
   - `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-execution-ledger-2026-07-09.md`
   - `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase6aa-svd-score-wiring-retry-result-2026-07-10.md`
   - `docs/plans/bayesfilter-multidim-lgssm-svd-score-wiring-demotion-result-2026-07-10.md`

3. Historical note, superseded for the current typed-identity lane: an earlier
   Phase 7 repair plan mentioned a public-API collection issue and tuning tools.
   The current frozen-transition serious path does not authorize retuning and
   has no unmatched collection prerequisite. Follow the operative Phase 7
   serious subplan and its declared focused/combined test gates instead.

4. If resuming LEDH score work, read these first:

   - `docs/plans/bayesfilter-ledh-score-wiring-repair-reset-memo-2026-07-10.md`
   - `docs/plans/bayesfilter-ledh-score-wiring-repair-visible-gated-execution-runbook-2026-07-10.md`
   - `docs/plans/bayesfilter-ledh-score-wiring-repair-visible-execution-ledger-2026-07-10.md`

5. Before any long run, write or verify an evidence contract with the exact
   baseline, promotion criterion, veto diagnostics, explanatory diagnostics,
   artifact path, and what the run will not prove.
