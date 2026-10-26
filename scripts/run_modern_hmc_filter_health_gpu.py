"""Bounded shared-GPU supervisor for the filter-health diagnostic.

Caller must use trusted GPU permissions. Scheduling never changes sampler
settings, retries numerical failures or migrates an active TensorFlow worker.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bayesfilter.runtime.gpu_provenance import query_nvidia_smi_gpus
from scripts.modern_hmc_campaign_status import tuning_outcome

PLAN = 'docs/plans/bayesfilter-modern-hmc-shared-gpu-continuation-2026-10-24.md'
PUBLIC_PLAN = 'docs/plans/bayesfilter-modern-hmc-public-gpu-continuation-2026-10-24.md'
RECOVERY_PLAN = 'docs/plans/bayesfilter-modern-hmc-public-gpu-recovery-2026-10-24.md'
BROAD_PLAN = 'docs/plans/bayesfilter-modern-hmc-broad-consumer-continuation-2026-10-25.md'
BROAD_RECOVERY_PLAN = 'docs/plans/bayesfilter-modern-hmc-broad-repair-2026-10-25.md'
REASSESS_SECONDS = 1800.0  # Owner instruction, not a worker timeout.
WAIT_CHECK_SECONDS = 60.0  # Operational cadence to catch shared-capacity openings.
MIN_FREE_MIB = 4096.0  # Inherited screen; actual peak remains to be measured.
MAX_LAUNCH_UTILIZATION = 50.0  # Owner: launch only below 50 percent.
HARDWARE = 'NVIDIA GeForce RTX 4080 SUPER'


def select_shared_gpu(rows, *, required_uuid=None):
    """Choose lowest load with headroom; busy processes are permitted."""
    candidates, identities, indices = [], set(), set()
    for raw in rows:
        row = dict(raw)
        index, uuid = row.get('nvidia_smi_index'), row.get('uuid')
        if type(index) is not int or not isinstance(uuid, str) or not uuid.startswith('GPU-'):
            raise ValueError('invalid physical GPU identity')
        if index in indices or uuid in identities:
            raise ValueError('duplicate physical GPU identity')
        indices.add(index); identities.add(uuid)
        if index not in (0, 1):
            continue
        if row.get('name') != HARDWARE:
            raise ValueError('GPU hardware class changed')
        for key in ('utilization_gpu_pct', 'memory_free_mib', 'memory_used_mib', 'memory_total_mib'):
            if isinstance(row.get(key), bool):
                raise ValueError('invalid numeric GPU telemetry')
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('invalid numeric GPU telemetry')
        if (row['utilization_gpu_pct'] > 100 or row['memory_total_mib'] <= 0
                or row['memory_free_mib'] + row['memory_used_mib'] > row['memory_total_mib']):
            raise ValueError('inconsistent GPU telemetry')
        if (required_uuid is not None and uuid != required_uuid):
            continue
        if (row['memory_free_mib'] >= MIN_FREE_MIB
                and row['utilization_gpu_pct'] < MAX_LAUNCH_UTILIZATION):
            candidates.append(row)
    return min(candidates, key=lambda r: (r['utilization_gpu_pct'],
        -r['memory_free_mib'], r['uuid']), default=None)


class Reassessment:
    """Record a fresh recommendation without changing the active device."""
    def __init__(self):
        self.last = None

    def due(self, now, *, waiting=False):
        interval = WAIT_CHECK_SECONDS if waiting else REASSESS_SECONDS
        return self.last is None or now - self.last >= interval

    def observe(self, now, rows, *, active_uuid=None, required_uuid=None):
        selected = select_shared_gpu(rows, required_uuid=required_uuid)
        if active_uuid is not None and active_uuid not in {r['uuid'] for r in rows}:
            raise ValueError('active GPU disappeared from trusted inventory')
        self.last = now
        return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
            gpu_inventory=list(rows), selected_for_next_worker=selected,
            active_uuid=active_uuid, required_uuid=required_uuid, active_worker_migrated=False,
            minimum_free_mib=MIN_FREE_MIB, reassess_seconds=REASSESS_SECONDS,
            wait_check_seconds=WAIT_CHECK_SECONDS,
            launch_utilization_below_percent=MAX_LAUNCH_UTILIZATION,
            sharing_permitted=True, trust_basis='trusted_escalated_gpu_execution')


def remaining_seconds(start, now, cap):
    return max(0.0, cap - (now - start))


def execution_remaining(start, worker_start, now, total_cap, worker_cap):
    remaining = remaining_seconds(start, now, total_cap)
    if worker_cap is not None:
        remaining = min(remaining, remaining_seconds(worker_start, now, worker_cap))
    return remaining


def reload_gpu_uuid(source):
    """Preserve the exported members' CUDA runtime identity on reload."""
    if source is None:
        raise ValueError('public reload requires source')
    manifest = json.loads((source/'manifest.json').read_text())
    uuid = manifest.get('environment', {}).get('CUDA_VISIBLE_DEVICES')
    if (not isinstance(uuid, str) or not uuid.startswith('GPU-')
            or ',' in uuid or manifest.get('physical_gpu', {}).get('uuid') != uuid
            or manifest.get('environment', {}).get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true'):
        raise ValueError('reload requires a recorded single physical GPU with memory growth')
    return uuid


def write(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def checkpoint_progress(output):
    """Report saved numerical work even if the worker's stage marker is stale."""
    path = Path(output)/'evidence/tuning/tuning_checkpoint.json'
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    result = payload['result']
    works = {w['work_item_id']: w for w in result.get('work_items', ())}
    unique_trials = set()
    for row in result.get('observations', ()):
        observation = row['observation']
        completed = observation.get('acceptance_evidence', {}).get('completed_trials', 0)
        work = works.get(row.get('work_item_id'), {})
        # Each observation includes all preceding trials in its evidence rung.
        # Count only this work item's newly completed interval, keeping fresh
        # verification and measurement streams separate. Partial chunks below
        # are explicitly chunks, not additional completed-trial observations.
        interval = work.get('trial_range') or observation.get('trial_range') or (0, completed)
        for ordinal in range(interval[0], min(completed, interval[1])):
            unique_trials.add((row.get('work_item_id'), ordinal))
    return dict(checkpoint=str(path), checkpoint_mtime_utc=datetime.fromtimestamp(
        path.stat().st_mtime, timezone.utc).isoformat(),
        complete_work_trials=len(unique_trials),
        trial_count_basis='unique_work_item_trial_ordinals_excluding_partial_chunks',
        partial_chunk_count=sum(map(len, payload.get('partial_chunks', {}).values())),
        completed_stage_counts={stage: sum(row.get('stage') == stage
            for row in result.get('observations', ())) for stage in ('measurement', 'verification')},
        candidate_states=result.get('candidate_states', {}),
        completion_status=result.get('completion_status'))


def cooperative_seconds(available, closeout):
    if not math.isfinite(closeout) or closeout < 10 or available <= closeout:
        raise ValueError('worker allowance must exceed a finite closeout reserve >= 10 seconds')
    return available-closeout


def stop_worker(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def supervise(args):
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 5:
        raise ValueError('finite wall cap must include more than 5 seconds for cleanup')
    worker_cap = getattr(args, 'worker_wall_seconds', None)
    if worker_cap is not None and (not math.isfinite(worker_cap)
            or worker_cap <= 10 or worker_cap > args.max_wall_seconds):
        raise ValueError('worker wall allowance must exceed 10 seconds and fit total cap')
    closeout = getattr(args, 'closeout_seconds', 10.)
    if getattr(args, 'task', 'filter-health') != 'filter-health':
        cooperative_seconds(worker_cap or args.max_wall_seconds, closeout)
    args.output.mkdir(parents=True, exist_ok=False)
    start, process, monitor = time.monotonic(), None, Reassessment()
    task=getattr(args,'task','filter-health')
    plan = (BROAD_RECOVERY_PLAN if task == 'broad-resume' else BROAD_PLAN if task.startswith('broad-') else RECOVERY_PLAN
            if task == 'public-resume' else PLAN if task == 'filter-health' else PUBLIC_PLAN)
    receipt = dict(status='selecting_device', plan=plan, command=[sys.executable, *sys.argv],
        started_utc=datetime.now(timezone.utc).isoformat(), max_wall_seconds=args.max_wall_seconds,
        reassess_seconds=REASSESS_SECONDS, numerical_worker_launched=False, active_run=True,
        wait_check_seconds=WAIT_CHECK_SECONDS,
        worker_wall_seconds=worker_cap, closeout_seconds=closeout,
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        observations=[], numerical_policy_unchanged=True, posterior_qualified=False)
    write(args.output/'manifest.json', receipt)
    def interrupted(signum, frame):
        raise InterruptedError('supervisor interrupted by signal '+str(signum))
    previous_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        required_uuid = reload_gpu_uuid(getattr(args, 'source', None)) if task in ('public-reload','public-resume','broad-reload','broad-resume') else None
        receipt['required_uuid'] = required_uuid
        while remaining_seconds(start, time.monotonic(), args.max_wall_seconds) > 5:
            now = time.monotonic()
            if worker_cap is not None and remaining_seconds(start, now, args.max_wall_seconds) < worker_cap:
                receipt['status'] = 'insufficient_worker_allowance_after_wait'
                return 1
            if monitor.due(now, waiting=True):
                observation = monitor.observe(now, query_nvidia_smi_gpus(), required_uuid=required_uuid)
                receipt['observations'].append(observation)
                write(args.output/'manifest.json', receipt)
                selected = observation['selected_for_next_worker']
                if selected is not None:
                    break
            time.sleep(min(1., remaining_seconds(start, time.monotonic(), args.max_wall_seconds)))
        else:
            receipt['status'] = 'budget_exhausted_waiting_for_capacity'
            return 1
        worker_start = time.monotonic()
        if worker_cap is not None and remaining_seconds(start, worker_start, args.max_wall_seconds) < worker_cap:
            receipt['status'] = 'insufficient_worker_allowance_after_wait'
            return 1
        receipt['capacity_wait_seconds'] = worker_start-start
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=selected['uuid'],
            TF_FORCE_GPU_ALLOW_GROWTH='true', TF_CPP_MIN_LOG_LEVEL='2',
            TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2', OMP_NUM_THREADS='2')
        command = [sys.executable, str(ROOT/'scripts/diagnose_modern_hmc_filter_health.py'),
            '--inputs', str(args.inputs.resolve()), '--output', str(args.output.resolve()/'evidence'),
            '--device', 'GPU', '--compile', '--compile-chain']
        if task != 'filter-health':
            mode=task.removeprefix('public-').removeprefix('broad-')
            command=[sys.executable,str(ROOT/'scripts/diagnose_modern_hmc_macrofinance_public.py'),
                '--inputs',str(args.inputs.resolve()),'--output',str(args.output.resolve()/'evidence'),
                '--device','GPU','--mode',mode,'--wall-seconds',
                str(cooperative_seconds(execution_remaining(start,worker_start,time.monotonic(),args.max_wall_seconds,worker_cap),closeout))]
            if mode in ('reload','resume'):
                if getattr(args,'source',None) is None:
                    raise ValueError('public reload/resume requires source')
                command.extend(['--source',str(args.source.resolve())])
            if mode == 'resume':
                command[1]=str(ROOT/'scripts/resume_modern_hmc_macrofinance_public.py')
                offset=command.index('--mode'); del command[offset:offset+2]
                offset=command.index('--device'); del command[offset:offset+2]
                command.extend(['--plan',plan])
            if task.startswith('broad-') and mode != 'resume':
                command[1]=str(ROOT/'scripts/continue_modern_hmc_macrofinance_broad.py')
                offset=command.index('--device'); del command[offset:offset+2]
                for field in ('design','price_source','gpu_evidence'):
                    value=getattr(args,field,None)
                    if value is not None: command.extend(['--'+field.replace('_','-'),str(value.resolve())])
        receipt.update(status='running', selected_uuid=selected['uuid'], worker_command=command,
            worker_environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH',
                'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS')})
        with (args.output/'worker.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            receipt.update(worker_pid=process.pid, numerical_worker_launched=True)
            write(args.output/'manifest.json', receipt)
            last_progress = None
            while process.poll() is None:
                now = time.monotonic()
                if last_progress is None or now-last_progress >= WAIT_CHECK_SECONDS:
                    progress = checkpoint_progress(args.output)
                    if progress is not None:
                        progress['observed_utc']=datetime.now(timezone.utc).isoformat()
                        write(args.output/'numerical-progress.json',progress)
                        receipt['numerical_progress']=progress
                        write(args.output/'manifest.json',receipt)
                    last_progress=now
                if execution_remaining(start, worker_start, now, args.max_wall_seconds, worker_cap) <= 5:
                    receipt['status'] = 'wall_budget_exhausted'
                    stop_worker(process)
                    break
                if monitor.due(now):
                    receipt['observations'].append(monitor.observe(now, query_nvidia_smi_gpus(),
                        active_uuid=selected['uuid'], required_uuid=required_uuid))
                    write(args.output/'manifest.json', receipt)
                time.sleep(1.)
        receipt['returncode'] = process.returncode
        progress = checkpoint_progress(args.output)
        if progress is not None:
            progress['observed_utc'] = datetime.now(timezone.utc).isoformat()
            write(args.output/'numerical-progress.json', progress)
            receipt['numerical_progress'] = progress
        if receipt['status'] == 'running':
            result_path = args.output/'evidence/result.json'
            result = json.loads(result_path.read_text()) if result_path.is_file() else {}
            if task in ('public-tune','public-resume','broad-tune','broad-resume'):
                receipt['tuning_outcome'] = tuning_outcome(result)
                status = receipt['tuning_outcome']['status']
                receipt['status'] = 'error' if status == 'completed' and process.returncode != 0 else status
            else:
                receipt['status'] = ('completed' if process.returncode == 0
                    and result.get('status') == 'completed' else 'repair_needed')
        elif receipt['status'] == 'wall_budget_exhausted':
            path = args.output/'evidence/tuning/tuning_checkpoint.json'
            if path.exists():
                receipt['tuning_outcome'] = tuning_outcome({}, checkpoint=json.loads(path.read_text()), wall_stopped=True)
        return 0 if receipt['status'] == 'completed' else 1
    except Exception as exc:
        receipt.update(status='error', error_type=type(exc).__name__, error=str(exc))
        return 1
    finally:
        stop_worker(process)
        receipt.update(elapsed_seconds=time.monotonic()-start,
            finished_utc=datetime.now(timezone.utc).isoformat(), active_run=False)
        write(args.output/'manifest.json', receipt)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        print(json.dumps({k:receipt[k] for k in ('status','elapsed_seconds','active_run')}), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-wall-seconds',type=float,required=True)
    parser.add_argument('--worker-wall-seconds',type=float,
        help='Reserve this full execution allowance before launch, within the inclusive wall cap')
    parser.add_argument('--task',choices=('filter-health','public-price','public-tune','public-reload','public-resume',
        'broad-price','broad-tune','broad-reload','broad-resume'),default='filter-health')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--design',type=Path)
    parser.add_argument('--price-source',type=Path)
    parser.add_argument('--gpu-evidence',type=Path)
    parser.add_argument('--closeout-seconds',type=float,default=10.,
        help='Reserve native-call completion and artifact writing before the hard cap')
    return supervise(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
