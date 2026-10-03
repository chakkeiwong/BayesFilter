#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# This wrapper is deliberately CPU-only.  It verifies the guide/policy repair;
# it is not a GPU HMC or posterior campaign.
export CUDA_VISIBLE_DEVICES="-1"
export TF_CPP_MIN_LOG_LEVEL="3"
python_bin="${PYTHON_BIN:-python}"

"${python_bin}" -m py_compile \
  bayesfilter/inference/fixed_transport_hmc_tuning_tf.py \
  bayesfilter/inference/fixed_transport_hmc_tuning.py \
  bayesfilter/inference/tuning_contract.py

"${python_bin}" scripts/render_hmc_tuning_interface_docs.py --check

"${python_bin}" -m pytest -q \
  tests/test_hmc_tuning_policy_repair.py \
  tests/test_hmc_tuning_documentation_contract.py
