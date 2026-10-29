#!/usr/bin/env bash
# Narrow launch surface for the reviewed September 30 campaign.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo 'Usage: run_neutra_warm_start_repair_campaign.sh start|status|check|configure' >&2
    exit 2
fi
cd /home/ubuntu/python/BayesFilter
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
case "$1" in
  configure)
    exec /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_neutra_warm_start_repair_master.py configure
    ;;
  check)
    export CUDA_VISIBLE_DEVICES=-1
    exec /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings \
      tests/test_neutra_warm_start_pipeline.py tests/test_neutra_warm_start_master.py \
      tests/test_neutra_warm_start_queue.py tests/test_neutra_warm_start_repair.py
    ;;
  start)
    if systemctl --user is-active --quiet neutra-warm-start-repair-20260930.service; then
      exec /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_neutra_warm_start_repair_master.py status
    fi
    exec systemd-run --user --unit=neutra-warm-start-repair-20260930 --collect \
      --property=WorkingDirectory=/home/ubuntu/python/BayesFilter \
      --setenv=TF_FORCE_GPU_ALLOW_GROWTH=true --setenv=XLA_PYTHON_CLIENT_PREALLOCATE=false \
      --setenv=TF_NUM_INTRAOP_THREADS=2 --setenv=TF_NUM_INTEROP_THREADS=2 \
      /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
      /home/ubuntu/python/BayesFilter/scripts/run_neutra_warm_start_repair_master.py run
    ;;
  status)
    exec /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_neutra_warm_start_repair_master.py status
    ;;
  *) echo 'Unknown campaign command' >&2; exit 2 ;;
esac
