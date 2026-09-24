#!/usr/bin/env bash
# LEDH while_loop refactor program: single-entry test/coverage/diagnostic runner.
#
# WHY THIS SCRIPT EXISTS: permission prefix matching treats compound commands
# (`cd X && VAR=Y python -m pytest ...`) as separate sub-commands, so allowlisting
# them individually produces a prompt per phase. Allowlisting this ONE script
# covers every command the program needs. Per the global cross-agent policy,
# narrow wrapper-script approval is preferred over broad `["python"]`/`["bash"]`.
#
# CPU-ONLY BY CONSTRUCTION: every mode below exports CUDA_VISIBLE_DEVICES=-1
# before Python starts, per the GPU/CUDA policy for deliberate CPU-only runs.
# GPU device-memory validation is NOT in this script; it is a separate deferred
# step requiring escalated permissions.
#
# Usage: bash scripts/run_phase_tests.sh <mode>
set -uo pipefail

REPO="/home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 2; }

# Deliberate CPU-only: set before any TF/JAX/PyTorch import.
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2

# Collection-error modules unrelated to the LEDH canonical route (Phase 0 audit).
IGNORES=(
  --ignore=tests/highdim/test_genut_shape_lm_tf.py
  --ignore=tests/highdim/test_zhao_cui_austria_sir_lane_b_t2_score_tf.py
)

FUSED="tests/highdim/test_ledh_canonical_batch_fused.py"
NONFUSED="tests/highdim/test_ledh_canonical_batch.py"
KERNEL="bayesfilter.highdim.ledh_canonical_batch_fused_tf"

MODE="${1:-}"

banner() { echo "=== [$MODE] $* ==="; }

case "$MODE" in
  # --- Phase 0 ---
  parity)
    banner "primary + non-fused parity gates (CPU-only)"
    python -m pytest "$FUSED" "$NONFUSED" -v --no-header
    ;;
  parity-fused)
    banner "primary parity gate only (CPU-only)"
    python -m pytest "$FUSED" -v --no-header
    ;;
  canonical)
    banner "canonical subset baseline (CPU-only, ~200s)"
    python -m pytest tests/highdim/ "${IGNORES[@]}" -k canonical -v --no-header
    ;;
  coverage)
    banner "baseline coverage of the refactor target kernel"
    python -m pytest "$FUSED" "$NONFUSED" \
      --cov="$KERNEL" --cov-report=term-missing --no-header
    ;;
  coverage-html)
    banner "coverage with HTML report"
    python -m pytest "$FUSED" "$NONFUSED" \
      --cov="$KERNEL" --cov-report=term-missing \
      --cov-report=html:htmlcov --no-header
    ;;
  install-cov)
    banner "install pytest-cov into the active conda env"
    python -m pip install pytest-cov
    ;;

  # --- Phase 1-3 regression + diagnostics ---
  score-suite)
    banner "score/UKF/tangent regression suite (refactor blast radius)"
    python -m pytest \
      tests/highdim/test_ledh_canonical_score_full.py \
      tests/highdim/test_ledh_canonical_score_stages.py \
      tests/highdim/test_ledh_canonical_score_step.py \
      tests/highdim/test_ledh_canonical_score_recursion.py \
      tests/highdim/test_ledh_canonical_score_ukf_tangent.py \
      tests/highdim/test_ledh_canonical_ukf_lifecycle.py \
      tests/highdim/test_ledh_canonical_neutra_target.py \
      -v --no-header
    ;;
  graph-size)
    banner "graph-node / GraphDef-bytes / trace-time / host-RSS diagnostic"
    python docs/benchmarks/diagnose_graph_size_20260830.py
    ;;
  eval-time)
    banner "trace+first vs warm eval-time diagnostic"
    python docs/benchmarks/diagnose_eval_time_20260830.py
    ;;
  direction-cost)
    banner "CSE-effectiveness / direction-cost-scaling diagnostic"
    python docs/benchmarks/diagnose_direction_cost_scaling_20260830.py
    ;;

  # --- Phase 4 ---
  surrogate-hmc)
    banner "surrogate-force HMC integration smoke (CPU-only)"
    python docs/benchmarks/step1_true_surrogate_force.py
    ;;
  env)
    banner "environment provenance for the run manifest"
    echo "conda env : ${CONDA_DEFAULT_ENV:-unset}"
    echo "python    : $(command -v python)"
    python - <<'PY'
import sys, tensorflow as tf
print("python ver:", sys.version.split()[0])
print("tf ver    :", tf.__version__)
print("tf gpus   :", tf.config.list_physical_devices("GPU"), "(CPU-only run: expected [])")
try:
    import tensorflow_probability as tfp
    print("tfp ver   :", tfp.__version__)
except Exception as exc:
    print("tfp ver   : import failed:", exc)
PY
    echo "git branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "git commit: $(git rev-parse HEAD)"
    ;;

  *)
    cat <<'USAGE'
Usage: bash scripts/run_phase_tests.sh <mode>

Phase 0:  parity | parity-fused | canonical | coverage | coverage-html | install-cov
Phase 1-3: score-suite | graph-size | eval-time | direction-cost
Phase 4:  surrogate-hmc
Any time: env

All modes are CPU-only (CUDA_VISIBLE_DEVICES=-1 exported before Python starts).
USAGE
    exit 2
    ;;
esac
