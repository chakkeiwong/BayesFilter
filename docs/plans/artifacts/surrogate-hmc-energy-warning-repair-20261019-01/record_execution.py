"""Record the controller repair and live restart; no numerical computation."""
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[4]
D = Path(__file__).resolve().parent
E = ROOT / 'docs/plans/artifacts/surrogate-hmc-48h-extension-20261018-01'
R = E / 'run'
def read(path):
    return json.loads(path.read_text())
state = read(R / 'state.json')
receipt = read(R / 'queue-energy-repair-01/handoff.json')
budget = read(R / 'budget.json')
active = R / 'range_bearing-energy-resume-01/range_bearing/inference'
manifest = read(active / 'manifest.json')
progress = read(active / 'public-tuning/fresh-progress.json')
preflight = read(D / 'preflight-result.json')
assert state['status'] == 'RUNNING' and state['current_model'] == 'range_bearing'
assert receipt['status'] == 'TAKEN_OVER'
assert progress['status'] == 'RUNNING' and progress['counts']['warmup'] >= 1500
assert preflight['status'] == 'PASS' and preflight['frozen_config_unchanged']
assert manifest['device'] == 'GPU' and manifest['particles'] == 4032
assert manifest['filter_dtype'] == 'float32' and not manifest['tf32_effective']
assert manifest['memory_policy']['all_physical_devices_memory_growth']
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
paths = [
    'bayesfilter/inference/hmc_tensorflow_verified.py',
    'bayesfilter/inference/hmc_tensorflow_resume.py',
    'scripts/run_surrogate_hmc_48h_queue_repair.py',
    'tests/test_hmc_energy_warning.py',
    'tests/test_neural_force_hmc.py',
    'tests/test_surrogate_hmc_48h_queue_repair.py',
]
after = D / 'source-after'
after.mkdir(exist_ok=True)
diff = []
for rel in paths:
    src = ROOT / rel
    shutil.copy2(src, after / src.name)
    before = D / 'source-before' / src.name
    if before.exists():
        diff.extend(difflib.unified_diff(before.read_text().splitlines(True),
            src.read_text().splitlines(True), fromfile='before/' + rel, tofile='after/' + rel))
(D / 'controller-repair.patch').write_text(''.join(diff))
record = dict(schema='surrogate_hmc_energy_warning_repair_execution_v1',
    recorded_utc=now, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
    command='/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_48h_queue_repair.py launch --resume-range-bearing',
    environment=manifest['python'], cpu_gpu_status='GPU; escalated launch; memory growth verified',
    memory_policy=manifest['memory_policy'], data_sha256=manifest['scope']['data_sha256'],
    random_seeds=progress['used_seeds'], wall_time='RUNNING; final worker time is recorded by the shared budget ledger',
    deadline_utc=budget['deadline_utc'], used_worker_seconds_before_current_attempt=budget['used_seconds'],
    remaining_worker_seconds_before_current_attempt=budget['remaining_seconds'],
    attempts_used=len(budget['attempts']), maximum_attempts=budget['max_attempts'],
    supervisor_pid=state['pid'], active_master_pid=state['active_master_pid'],
    progress=progress, plan='docs/plans/surrogate-hmc-energy-warning-repair-2026-10-19.md',
    result=str(D / 'execution.md'), output=str(active), receipt=str(R / 'queue-energy-repair-01/handoff.json'),
    test_evidence={'controller_resume_queue':'final-focused-tests.log: 78 passed',
       'updated_queue':'queue-restart-tests.log: 23 passed (replaces the 22-test queue subset)',
       'mh':'mh-tests.log: 5 passed', 'unique_total':84},
    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
(D / 'repair-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
note = f'''# Executed finite-energy warning repair

Recorded {now}. Range bearing is RUNNING from the saved state: {progress['counts']['warmup']}
warmup transitions per chain, {progress['counts']['verification']} retained draws at this check. The
worker log confirms saved-state replay, and the original GPU was initialized.
This is a live restart, not a completed sampling result.

Large finite energy errors are warnings in fresh and resumed chunks and final
admission checks. Proposal dynamics and the MH acceptance test are unchanged.
The finite energy threshold remains 1000 only for reporting. Genuine arithmetic
and nonfinite failures, invalid endpoints, corruption and lack of chain movement
remain separate checks. The historical energy veto is available only when
validating an old archive and cannot control new sampling.

The saved archive passes exact original-GPU replay: six chunks, 1500 warmup
draws per chain, unchanged states, seeds, frozen configuration and numerical
kernel hashes. The old energy-only stop is preserved in its original directory;
its copied history now emits an energy warning and permits continuation.
An empty-phase reader mismatch was also repaired; both null and empty-object
representations work, while missing populated aggregates still fail.

The focused evidence comprises 84 passing tests. Logs and before/after source
copies accompany repair-manifest.json. CPU archive replay exposed last-bit
device differences; no equality tolerance was relaxed, and GPU replay passed.

The first handoff command found that the predecessor had already finished and
correctly refused to attach. The idle-restart branch was then tested and launched
successfully. Predator-prey had completed VERIFIED. SIR's immediate empty-phase
reader failure is repaired, and its retry follows range bearing automatically.

The queue preserves the original 172800 worker-second, six-attempt budget and
deadline {budget['deadline_utc']}. The range-bearing restart has an eight-hour
cap; SIR receives the remaining budget in a fresh directory. Five attempts have
now been started, including this active attempt. No new budget was granted.

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Resume saved range-bearing kernel | Regression tests and real archive replay pass | No non-energy replay veto | Future mixing and cost | Continue existing master automatically | Convergence or posterior accuracy |

| Inference status | Finding |
|---|---|
| Hard veto screen | Archived non-energy health checks pass; new chunks remain subject to those checks |
| Statistically supported ranking | None; this is a controller repair |
| Descriptive-only differences | Historical energy veto is now a warning; no new performance comparison |
| Default-readiness | Not assessed; owner-directed stopping policy implemented |
| Next evidence needed | Actual cumulative retained-chain diagnostics under the frozen kernel |

Post-run red-team limitation: replay and tests establish continuation mechanics,
not eventual mixing. Many rejections could still leave insufficient effective
samples; cumulative diagnostics and limits remain active. The finite endpoint
approximation is unchanged, and exact-posterior agreement is not established.
'''+ '\nActual launch command:\n```sh\n'+record['command']+'\n```\n'
(D / 'execution.md').write_text(note)
checkpoint = ROOT / 'docs/plans/surrogate-hmc-performance-repair-checkpoint-2026-10-11.md'
shutil.copy2(checkpoint,D / 'checkpoint-before.md')
checkpoint.write_text(f'''# Active surrogate-HMC checkpoint

Updated {now}. Goal: complete the five-model surrogate-HMC campaign.
Current owner direction: finite energy error alone must not stop sampling;
repair the master and resume range bearing. Repair executed; campaign incomplete.

Checkout: {ROOT}
Branch: {record['branch']}; HEAD {record['git_commit']}.
Preserve unrelated dirty changes. GPU commands require escalation; verified
memory growth is mandatory. Python: /home/chakwong/anaconda3/envs/tftwogpu/bin/python.

Active plan: docs/plans/surrogate-hmc-energy-warning-repair-2026-10-19.md.
Evidence D: docs/plans/artifacts/surrogate-hmc-energy-warning-repair-20261019-01.
Campaign E: docs/plans/artifacts/surrogate-hmc-48h-extension-20261018-01.
E/run/budget.json is the shared ledger: 172800 worker seconds, maximum six
attempts, deadline 2026-10-20 08:00:41 UTC (20 October 16:00:41 Hong Kong).
Five attempts started, one still running. Before this attempt, charged time was
{budget['used_seconds']:.2f}s and remaining worker allowance {budget['remaining_seconds']:.2f}s;
the absolute deadline also applies. Never infer remaining budget from stale aliases.

Active supervisor PID {state['pid']}; range master PID {state['active_master_pid']};
GPU worker PID 4166676 was confirmed alive. Recheck PIDs before acting.
Receipt E/run/queue-energy-repair-01/handoff.json is TAKEN_OVER.
New range output E/run/range_bearing-energy-resume-01/range_bearing/inference.
Fresh progress reports RUNNING, {progress['counts']['warmup']} warmup and
{progress['counts']['verification']} retained per chain after exact archive replay.
The next new chunk is being executed; no completed new chunk is claimed yet.

Repaired modules: hmc_tensorflow_verified.py (warning-only finite energy),
hmc_tensorflow_resume.py (exact known source migration, energy-only resumption,
empty-phase reader), run_surrogate_hmc_48h_queue_repair.py (idle queue restart).
Numerical proposal/MH kernel, target, data, priors and seeds are unchanged.
84 focused tests pass, including actual MH rejection, plus six-chunk exact GPU
archive replay. D contains source-before/after, patch, logs and execution manifest.
No energy stop remains in the active sampler; historical gates are replay-only.

Predator-prey completed VERIFIED at E/run/predator_prey-retry-01. SIR's prior
attempt stopped on the empty-phase schema mismatch before sampling; it will
retry automatically at E/run/sir_d18-energy-retry-01 after range bearing.
Do not launch a duplicate. Range has an eight-hour cap, SIR the remaining budget.

Frozen settings: N4032/K2016, FP32/TF32off, GPU/XLA, epsilon1/five OT iterations,
literature model priors. T50 except range T20 with16 flow steps. Four chains:
warmup min2000, latest1000 modern R-hat<=1.05, max10000; retained cumulative
modern R-hat<=1.01, max20000 or budget. No extra validation draws.

Exact next action: inspect E/run/state.json, budget.json, next-phase-plan.json,
and active range public-tuning/fresh-progress.json. Confirm new chunks advance
beyond1500; let the master continue and advance to SIR automatically.
Only repair a new failure within the same contract and remaining budget.

Prior LGSSM and KSC sampling rule passed (R-hat1.00708 and1.00600). Full
posterior agreement remains not checked; no five-model completion or ranking
claim is established. Detailed earlier history is in D/checkpoint-before.md.
''')
print(json.dumps({'status':'recorded','manifest':str(D/'repair-manifest.json'),
    'checkpoint':str(checkpoint),'range_status':progress['status'],'counts':progress['counts']}))
