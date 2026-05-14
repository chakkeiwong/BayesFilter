# Supervisor audit: BayesFilter V1 Model B/C BC0-BC8 Subplans

## Date

2026-05-14

## Scope

Codex acted as supervisor for the plan-review cycle requested by the user.
Claude Code was used only as a bounded critical reviewer through:

```bash
bash scripts/claude_worker.sh --cwd /home/chakwong/BayesFilter --name <name> --model sonnet --effort high "<worker prompt>"
```

Claude was instructed not to edit files, not to run tests, not to run GPU/HMC
experiments, and to return `ACCEPT` or `REJECT`.

Reviewed artifacts:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
docs/plans/bayesfilter-v1-model-bc-bc0-baseline-reconciliation-plan-2026-05-14.md
docs/plans/bayesfilter-v1-model-bc-bc1-wider-branch-boxes-plan-2026-05-14.md
docs/plans/bayesfilter-v1-model-bc-bc2-score-accuracy-stress-plan-2026-05-14.md
docs/plans/bayesfilter-v1-model-bc-bc3-horizon-noise-robustness-plan-2026-05-14.md
docs/plans/bayesfilter-v1-model-bc-bc4-reference-decision-plan-2026-05-14.md
docs/plans/bayesfilter-v1-model-bc-bc5-hmc-ladder-plan-2026-05-14.md
docs/plans/bayesfilter-v1-model-bc-bc6-gpu-xla-scaling-plan-2026-05-14.md
docs/plans/bayesfilter-v1-model-bc-bc7-hessian-consumer-decision-plan-2026-05-14.md
docs/plans/bayesfilter-v1-model-bc-bc8-consolidation-release-gate-plan-2026-05-14.md
docs/source_map.yml
scripts/claude_worker.sh
```

## Codex local inspection

Before asking Claude, Codex inspected:

- the requested Model B/C master program;
- the prior V1 master program and execution summary;
- the prior V1 master/subplan audit;
- representative P1/P4/P6 subplans;
- planning templates;
- parent `AGENTS.md` policy;
- current test and plan filenames relevant to nonlinear B/C, HMC, GPU/XLA,
  branch diagnostics, and reset-memo/source-map conventions.

Initial Codex audit found that the master was directionally sound but needed
separate executable subplans with explicit evidence contracts, entry gates,
stop conditions, artifact names, and continuation rules.

## Claude round 1

Claude returned: `REJECT`.

Codex accepted the rejection as valid.  Must-fix blockers were:

- BC5/BC8 accidentally implied an HMC convergence ladder while the gates only
  supported HMC readiness classification.
- BC6 was unnecessarily ordered behind BC5 even though GPU/XLA scaling depends
  on stable shapes, not HMC classification.
- BC1-BC3 lacked bounded stop/narrowing rules.
- BC2 lacked predeclared tolerance rules.
- BC8 did not include branch-diagnostic coverage in final validation.
- Result artifact names used fixed drafting dates instead of execution-date
  placeholders.
- Reset-memo update language could be misread as plan-review work rather than
  execution-time work.
- BC4 needed a required per-claim comparator/reference mapping table.

## Revisions made

Codex revised the plans to:

- add BC0-BC8 subplan file paths to the master program;
- make BC5 an HMC readiness ladder, not a convergence ladder;
- declare BC5 and BC6 independent downstream diagnostics after BC0-BC4 gates;
- add BC1 stop/narrowing rules;
- add BC2 predeclared tolerance requirements;
- add BC3 horizon/noise stop rules;
- add a BC4 required comparator/reference decision table;
- add execution-date placeholder rules for result artifacts;
- include branch diagnostics in BC8 focused nonlinear validation;
- clarify BC0 reset-memo updates happen during execution, not plan review;
- register the BC subplans in `docs/source_map.yml`;
- add the repo-local Claude worker forwarding wrapper.

## Claude round 2

Claude returned: `ACCEPT`.

Claude's residual non-blocking risks:

- execute BC8 using the stricter BC8 subplan wording if the master and subplan
  differ on opt-in HMC diagnostics;
- the grouped source-map entry is acceptable now but may be split later if
  per-phase provenance becomes necessary.

## Codex final audit

Decision: `ACCEPT`.

The BC0-BC8 subplans are good to go as planning artifacts.  They are ready for
future execution under the master cycle:

```text
plan -> skeptical audit -> execute -> test -> audit result -> tidy -> reset-memo update -> result artifact -> continuation decision
```

Execution must still stop if any phase primary gate fails or a veto diagnostic
fires.

## Verification

Local checks run during this planning pass:

```bash
bash -n scripts/claude_worker.sh
```

Observed: passed.

```bash
python - <<'PY'
from pathlib import Path
import yaml
with Path("docs/source_map.yml").open(encoding="utf-8") as fh:
    yaml.safe_load(fh)
print("parsed docs/source_map.yml")
PY
```

Observed: parsed `docs/source_map.yml`.

Markdown trailing-whitespace check over the master and BC0-BC8 subplans:

```text
trailing_ws=[]
```

No numerical tests, GPU commands, HMC diagnostics, or long experiments were run
in this planning pass.

## Remaining unproven

- No Model B/C branch box, score stress, robustness, reference, HMC, GPU/XLA,
  or Hessian claim has been newly tested by this pass.
- Claude's `ACCEPT` is not evidence of numerical correctness.
- The subplans are execution-ready, but each future phase still requires its
  own evidence contract, run manifest, result artifact, and reset-memo update.
