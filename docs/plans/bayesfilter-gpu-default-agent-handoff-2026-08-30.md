# Handoff memo: making GPU execution the BayesFilter default

**Audience:** the next coding or operations agent working on this repository  
**Date:** 2026-08-30  
**Purpose:** explain exactly what has been implemented, what remains owned by
the outer Codex/service gateway, and how to verify the boundary without adding
a one-by-one idle-GPU or Luna approval step.

## The short answer

GPU-default behavior has two separate owners:

1. **The repository** chooses GPU 0, enables TensorFlow on-demand allocation
   before import, rejects an invalid GPU setup, and writes provenance.
2. **The outer Codex/service gateway** decides whether a process is allowed to
   see `/dev/nvidia*` and whether a command needs approval.

Repository code cannot grant itself the second permission. The correct service
configuration is one narrow, persistent, project-scoped rule for the stable
GPU launcher, while unrelated commands remain under the normal approval
policy. Do not use a global `approval_policy = "never"`, a danger-full-access
bypass, or a broad rule for all shell/Python commands merely to make this
repository convenient.

### Copy-paste instructions for the next agent

> Read this memo and the boundary note first. The repository-side default is
> already implemented. Ask the service owner for one persistent project-scoped
> permission for the exact absolute launcher command, with GPU 0 exposed; do
> not add an idle-GPU probe, invoke Luna per retry, or put `codex exec` inside
> the launcher. Then run the launcher directly and verify its manifest. If the
> process cannot see a device before the runner starts, report an outer
> service-boundary failure rather than changing the sampler or adding a CPU
> fallback.

## What is already in the repository

The canonical entry point is:

```text
/home/ubuntu/python/BayesFilter/scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh
```

Run it from any directory with the stable absolute command:

```bash
bash /home/ubuntu/python/BayesFilter/scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh
```

Or, from the repository root:

```bash
cd /home/ubuntu/python/BayesFilter
bash scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh
```

The launcher deliberately does the following before the Python runner imports
TensorFlow:

- defaults `CUDA_VISIBLE_DEVICES` to physical GPU `0`;
- exports `TF_FORCE_GPU_ALLOW_GROWTH=true`;
- uses the `tfgpu` Python environment unless `PYTHON_BIN` is explicitly set;
- creates a fresh, timestamped output directory and refuses to overwrite one;
- invokes the repository Phase 8 runner with XLA/TF32 and the repository
  memory-policy checks; and
- does **not** call `codex-gpu-probe`, an idle-GPU probe, or a per-run Luna
  reviewer.

The Python runner then verifies the actual visible device and memory policy.
The launcher setting is not sufficient by itself: a run is valid only if the
manifest confirms that TensorFlow saw the device and growth was enabled before
logical-device initialization.

## The one-time service change the other agent must request

The service administrator or gateway owner must configure a rule conceptually
equivalent to:

```text
project/workdir: /home/ubuntu/python/BayesFilter
allowed command: bash /home/ubuntu/python/BayesFilter/scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh
device: expose GPU 0 to this project/session
approval: persistent for this exact project command
all other commands: retain the normal on-request approval policy
```

The exact syntax is service-specific and must be entered in the service's own
configuration UI or policy store; it is not a repository file and cannot be
fixed by editing `AGENTS.md` or a Python script. If the service matches an
exact argument vector, use the stable absolute command above when creating the
rule. Keep optional overrides out of the saved rule unless each override is
intentionally part of the approved command.

Do not try to solve this by nesting a `codex` invocation inside the shell
launcher. The outer Codex/service process is the authorization boundary; a
nested invocation only creates another service request and can reintroduce the
same prompt or gateway failure.

A useful request to the gateway owner is:

> For `/home/ubuntu/python/BayesFilter`, expose physical GPU 0 and persist
> approval for the exact repository launcher command above. Keep normal
> approval for unrelated commands. Do not require an idle-GPU probe or a Luna
> review on every retry. Do not grant global shell, Python, or danger-full
> access.

This is a service boundary change, not a scientific claim. Project trust in a
Codex configuration does not necessarily expose device files; in the current
managed workspace the non-elevated sandbox has no `/dev/nvidia*`. A service may
instead designate the session as trusted managed GPU execution. Only the
service may truthfully provide that designation. The repository may record

```text
BAYESFILTER_GPU_TRUST_BASIS=owner_designated_managed_session_visible_gpu_trusted
```

when the service actually supplies it. An agent must never set that value just
to relabel an unverified run.

## What the local Codex configuration means

The local configuration currently uses normal on-request approvals and marks
this project as trusted. That is useful for repository work, but it does not
override the service sandbox, attach a GPU, or create a persistent command rule
for every gateway. The next agent should therefore leave the local setting
alone and ask for the project-scoped service rule described above. Changing the
global setting to `never` would be a security-policy change with no scientific
benefit and would make unrelated commands harder to control.

## Verification procedure

After the service rule is installed, perform a cheap boundary check before any
long experiment.

1. Launch the stable command once. It should start without an idle-probe or
   Luna approval prompt, and it should either produce a manifest or a clear
   device/setup error. Do not interpret a timeout as a scientific result.
2. Inspect the newest run manifest and confirm all of these facts:

   ```text
   gpu_environment.cuda_visible_devices == "0"
   logical_gpus contains exactly one GPU
   TF_FORCE_GPU_ALLOW_GROWTH == "true"
   memory_policy reports growth verified for every visible physical GPU
   gpu_launch_mode == "repository_default_gpu_launcher"
   external_approval_is_runner_gate == false
   gpu_trust_basis matches what the service actually provided
   ```

3. Check the launcher source if needed:

   ```bash
   rg -n "codex-gpu-probe|idle|approval|CUDA_VISIBLE_DEVICES|TF_FORCE_GPU_ALLOW_GROWTH" \
     /home/ubuntu/python/BayesFilter/scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh
   ```

   The expected result is the two environment-setting lines and no probe or
   approval invocation.

4. Preserve the manifest, command, environment, seed, device information, and
   output directory in the experiment note. A successful boundary check proves
   execution access and allocator setup; it does **not** prove whitening,
   posterior correctness, HMC readiness, or transport quality.

## How to classify failures

| Observation | Correct interpretation | Next action |
|---|---|---|
| No `/dev/nvidia*`, TensorFlow lists no GPU, or device initialization is denied before the runner starts | Outer service/device boundary failure | Ask the gateway owner to expose GPU 0 or apply the narrow project rule. Do not add a CPU fallback or relabel the run. |
| Luna/reviewer returns 404/502/503 or the approval service is unavailable before the process starts | Approval-service infrastructure failure | Use the persistent project rule or a service with managed GPU access. This is not evidence about the sampler or model. |
| The runner reaches GPU 0, writes a `timeout.json`, and exits 124 | Application graph/compile/cost feasibility failure | Read the timeout/postmortem; preserve it. It is not an approval failure and is not a candidate-quality result. |
| The runner writes `failure.json` for non-finite values, invalid checkpoints, or failed memory policy | Numerical or implementation failure | Repair the named code path under a new bounded attempt; do not weaken the gate. |
| A complete manifest passes all declared checks | GPU execution boundary passed | Continue only under the reviewed scientific plan. Do not infer a posterior or HMC claim from the manifest alone. |

## Current experiment status

The execution boundary has already been exercised three times through the
repository launcher. Each attempt reached GPU 0 without an idle-GPU probe or a
per-run Luna review, and TensorFlow initialized an RTX 4080 SUPER with XLA and
on-demand allocation. The relevant boundary record is
`docs/plans/bayesfilter-gpu-default-execution-boundary-2026-08-29.md`.

The active Phase 8 C1 experiment is a separate issue. Its authorized cost
budget is exhausted after repeated q=20 graph-cost timeouts; C2--C5 were not
started and no transport, whitening, posterior, or HMC candidate was promoted.
See
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-phase8-c1-result-2026-08-30.md`.
Do not rerun the expensive pilot simply to test approval. A new run needs a new
reviewed cost/graph plan and output root (or an explicitly authorized new
budget). The launcher default is a repository execution default, not permission
to exceed the closed scientific campaign budget.

The follow-on bounded graph-repair diagnostic also reached GPU 0. Its `B=8`
chunked prefix passed (32 finite rows in 192.694 seconds), but the full 256-row
bank exceeded the new 600-second repair cap. This confirms that the gateway is
no longer the blocker; the remaining blocker is q=20 target graph cost. The
repair result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-graph-repair-result-2026-08-30.md`.

## Common mistakes to avoid

- **Editing repository policy to solve a gateway problem.** `AGENTS.md`, the
  runner, and the launcher can state and verify the default, but only the
  outer service can expose a device or persist approval.
- **Following a legacy approval wrapper.** Several older leaderboard scripts
  in `scripts/` deliberately require one-time human approval records. They are
  historical routes for a different campaign, not the Phase 8 repository GPU
  default. Use only
  `run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh` for this handoff.
- **Using an idle-GPU probe as a hidden scheduler.** It adds a per-run gate and
  does not establish TensorFlow memory growth. The launcher intentionally omits
  it.
- **Using a global approval bypass.** `approval_policy=never`, broad `bash` or
  `python` permission, and danger-full-access remove protections unrelated to
  this project and are not required.
- **Setting the managed-trust environment variable by hand.** It is
  provenance, not authorization. Leave the unclassified value when the service
  has not designated trusted managed GPU execution.
- **Assuming a visible GPU means enough memory or a completed experiment.**
  Growth is on-demand, not a hard cap; the run still needs its own cost and
  scientific gates.
- **Calling the launcher with `./...` without checking its mode.** The file is
  intentionally invoked with `bash`; use the absolute command above. Optional
  mode, backend, timeout, and validation-size overrides create a different
  experiment and must remain within a reviewed plan.

## Handoff checklist

The next agent can consider the gateway part complete only when:

- the service owner has installed the narrow persistent project rule;
- the stable launcher starts without a per-run idle/Luna gate;
- a fresh manifest records one visible GPU, pre-import growth verification, and
  the actual trust basis; and
- the boundary result is recorded separately from the Phase 8 scientific result.

Once those conditions hold, leave the repository launcher unchanged and work on
the declared Phase 8 graph-cost blocker under a new scientific plan. Do not
reopen the approval design or claim that the GPU default solves the q=20
transport feasibility problem.
