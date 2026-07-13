# LEDH TF32-Only Execution Policy Reset

date: 2026-07-10
status: ACTIVE_OWNER_DIRECTIVE

## Directive

For LEDH / LEDH-PFPF-OT work, production, admission, score, leaderboard, and
default evidence runs must use TensorFlow `float32` tensors with TensorFlow TF32
execution enabled.

Concrete CLI/settings:

- `--dtype float32`
- `--tf32-mode enabled`
- TensorFlow tensors use `tf.float32`
- TensorFlow TF32 execution is enabled before the LEDH computation

Terminology: TF32 is not a TensorFlow tensor dtype. In this repository, "TF32
LEDH" means `float32` tensors with TensorFlow TF32 execution enabled.

## Superseded Use

Do not use `dtype=float64` LEDH runs as current production, score-admission,
leaderboard, or default-policy evidence.

Existing historical artifacts that used float64 remain factual records of what
was run, but they are demoted to historical/reference diagnostics under the
current owner directive. In particular:

- `docs/plans/ledh-phase5-lgssm-score-memory-n10000-2026-07-06.json`
  records an LGSSM score-memory diagnostic with `dtype=float64`; it must not be
  cited as current TF32 LEDH production/admission evidence.

## Required Future Behavior

Future LEDH score/value diagnostics should fail closed if a production/admission
run attempts to use `dtype=float64`.

Allowed non-production mentions of older float64 LEDH artifacts:

- historical traceability;
- explaining why a previous result is no longer admissible;
- independent non-LEDH reference math, if clearly separated from LEDH execution.

Forbidden claims/actions:

- Do not call a float64 LEDH artifact current production evidence.
- Do not use float64 same-scalar FD as a promotion gate for LEDH production.
- Do not repair an LEDH score path by switching to float64.
- Do not report "TF32 dtype"; report `float32` dtype with TF32 execution enabled.

## Immediate Consequence

The recent value-stage finding remains valid because it used `float32` with TF32
enabled:

- `docs/plans/bayesfilter-ledh-lgssm-particle2500-t10-t50-value-ladder-result-2026-07-10.md`

The recent score blocker remains valid because it also used `float32` with TF32
enabled:

- `docs/plans/bayesfilter-ledh-lgssm-particle2500-score-only-ladder-blocker-result-2026-07-10.md`

The next score repair must target the TF32 LEDH route directly, not a float64
diagnostic route.
