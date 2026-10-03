# BayesFilter GPU default execution boundary

Date: 2026-08-29 (updated 2026-08-30)  
Status: `REPOSITORY_DEFAULT_IMPLEMENTED_SERVICE_RULE_VERIFIED_C1_BUDGET_SEPARATE`

BayesFilter GPU execution is a repository default. The active Phase 8 route
uses `scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh`, which
selects GPU 0 by default, sets `TF_FORCE_GPU_ALLOW_GROWTH=true` before any
TensorFlow import, refuses to overwrite a run directory, and delegates
numerical, XLA, memory-growth, and artifact checks to the Python runner. It
does not call `codex-gpu-probe`, an idle-GPU probe, or a per-run Luna reviewer.

This repository decision is separate from the Codex execution boundary. The
current `workspace-write` sandbox has no `/dev/nvidia*` devices, so a process
launched in that sandbox cannot establish GPU evidence. The outer service must
provide either a trusted managed-session GPU context or a narrow persistent
permission for this launcher. Setting a global `approval_policy = "never"` or
using a danger-full-access bypass would also remove protections for unrelated
commands and is not an appropriate repository default.

The runner records `external_approval_is_runner_gate=false` and records the
actual trust basis supplied by the service. If the service refuses the direct
launcher, that is platform evidence only; no indirect GPU path, hidden CPU
fallback, or relabeling of a CPU run as GPU evidence is allowed.

## One-time service configuration

Configure the service outside the repository with a project-scoped rule that
permits the exact launcher command and exposes the selected GPU. The rule must
leave unrelated commands under the normal approval policy. After that change,
the only repository command is:

```text
bash scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh
```

The service configuration is not committed here because it is account- and
administrator-owned. Its successful activation is demonstrated by the runner's
manifest fields `gpu_environment`, `gpu_trust_basis`, and `memory_policy`, not
by a self-attested approval token.

If the service designates the session as trusted managed GPU execution, it may
inject `BAYESFILTER_GPU_TRUST_BASIS=owner_designated_managed_session_visible_gpu_trusted`
for this project. The launcher otherwise records
`repository_default_gpu_route_external_boundary_unclassified`, which is the
honest value when the outer service has not supplied that designation. This
environment field is provenance only; it does not grant device access or
replace the service's one-time project rule.

## Boundary check

The first invocation after this route was installed used the launcher directly
and was admitted to GPU 0 without an idle-GPU probe or a per-run Luna review.
TensorFlow initialized one RTX 4080 SUPER with XLA and on-demand allocation;
the process later reached the q20 beta-0 checkpoint before the independent
1,800-second experiment cap. The timeout is recorded in the Phase 8 attempt
directory and says nothing about the service's GPU availability or the
transport candidate. It confirms the repository/service separation: the
launcher owns the scientific GPU defaults, while the outer service owns only
whether that process may access a device.

The subsequent bounded localization and small-bank retry were also admitted by
the same launcher without an idle-probe or per-run Luna gate. They reached GPU 0
and produced the timeout records described in the Phase 8 C1 result. The
service boundary is therefore operationally verified for this repository route;
the remaining C1 stop is a q=20 graph-cost budget veto, not an approval-policy
failure.

## Independent boundary probe (2026-08-30)

After the service switch, a read-only `nvidia-smi` check saw both RTX 4080 SUPER
devices. The repository TensorFlow allocator probe was then run with the stable
UUID for physical GPU 0:

```text
PYTHONPATH=/home/ubuntu/python/BayesFilter \
CUDA_VISIBLE_DEVICES=GPU-a1ea1946-07c0-8ed5-2ba1-d96f82c89cd3 \
TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 \
timeout 120s /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
scripts/run_tensorflow_gpu_probe.py \
--output docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/gpu-boundary/tensorflow-gpu-probe.json
```

The probe status is `pass`: TensorFlow 2.20.0 saw one logical GPU, the matrix
operation ran on `/device:GPU:0`, and the memory-policy receipt reports
`all_physical_devices_memory_growth=true` and
`configured_before_logical_device_initialization=true`. The preserved JSON is
the bounded infrastructure artifact; it carries no filtering, transport,
HMC, convergence, or performance claim. A first direct invocation without
`PYTHONPATH` failed before TensorFlow import because the standalone script does
not add the repository root to `sys.path`; the explicit-path retry repaired that
diagnostic-only invocation issue.
