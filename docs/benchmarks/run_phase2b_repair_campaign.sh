#!/usr/bin/env bash
# Phase 2B repair campaign: execute 9 missing runs
#
# Missing runs:
# - ksc_sv_T10 sobol_matousek: seeds 98301, 98302, 98303
# - predator_prey_T20 halton_owen: seeds 98301, 98302, 98303
# - predator_prey_T20 genut_guided: seeds 98301, 98302, 98303

set -euo pipefail

ARTIFACT_ROOT="docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904"
RUNNER="docs/benchmarks/run_rqmc_phase2b_repair.py"

# Use GPU 1 (4080 SUPER)
export CUDA_VISIBLE_DEVICES=1

echo "========================================================================"
echo "Phase 2B Repair Campaign"
echo "========================================================================"
echo "Runs: 9"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Conda env: tftwogpu"
echo ""

run_count=0
success_count=0
failure_count=0

run_one() {
    local model=$1
    local arm=$2
    local seed=$3
    local tuning_artifact=$4

    run_count=$((run_count + 1))
    run_id="${model}_${arm}_seed${seed}"
    output_dir="${ARTIFACT_ROOT}/runs/${run_id}"
    log_file="${ARTIFACT_ROOT}/logs/${run_id}_phase2b_repair.log"

    echo "----------------------------------------"
    echo "Run ${run_count}/9: ${run_id}"
    echo "----------------------------------------"

    if conda run -n tftwogpu python "${RUNNER}" \
        --model "${model}" \
        --arm "${arm}" \
        --seed "${seed}" \
        --tuning_artifact "${tuning_artifact}" \
        --output "${output_dir}" \
        > "${log_file}" 2>&1; then

        echo "  ✓ SUCCESS"
        success_count=$((success_count + 1))
    else
        echo "  ✗ FAILED (see ${log_file})"
        failure_count=$((failure_count + 1))
    fi
    echo ""
}

# KSC SV sobol_matousek (3 runs)
KSC_TUNING="docs/benchmarks/artifacts/ledh_trust_region_ksc_sv_t10_20260903/result.json"
run_one "ksc_sv_T10" "sobol_matousek" 98301 "${KSC_TUNING}"
run_one "ksc_sv_T10" "sobol_matousek" 98302 "${KSC_TUNING}"
run_one "ksc_sv_T10" "sobol_matousek" 98303 "${KSC_TUNING}"

# Predator-Prey halton_owen (3 runs)
PP_TUNING="docs/benchmarks/artifacts/ledh_trust_region_predator_prey_t20_20260903/result.json"
run_one "predator_prey_T20" "halton_owen" 98301 "${PP_TUNING}"
run_one "predator_prey_T20" "halton_owen" 98302 "${PP_TUNING}"
run_one "predator_prey_T20" "halton_owen" 98303 "${PP_TUNING}"

# Predator-Prey genut_guided (3 runs)
run_one "predator_prey_T20" "genut_guided" 98301 "${PP_TUNING}"
run_one "predator_prey_T20" "genut_guided" 98302 "${PP_TUNING}"
run_one "predator_prey_T20" "genut_guided" 98303 "${PP_TUNING}"

echo "========================================================================"
echo "Phase 2B Repair Campaign Complete"
echo "========================================================================"
echo "Total runs: ${run_count}"
echo "Success: ${success_count}"
echo "Failures: ${failure_count}"
echo ""

if [ ${failure_count} -eq 0 ]; then
    echo "✓ All Phase 2B repairs completed successfully"
    exit 0
else
    echo "✗ ${failure_count} runs failed"
    exit 1
fi
