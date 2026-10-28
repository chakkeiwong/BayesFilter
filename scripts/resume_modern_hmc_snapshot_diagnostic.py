#!/usr/bin/env python3
"""Host-only recovery of a frozen modern-HMC diagnostic.

The numerical implementation and checkpoint remain owned by the original
snapshot. This separately recorded entrypoint supplies the finite live budget
check missing from the historical frozen harness; it never edits that harness.
Run with trusted GPU permissions and an explicit inclusive allowance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_runtime(campaign):
    source = read(campaign/'isolated-source.json')
    root, consumer = Path(source['source_root']), Path(source['consumer_root'])
    for name in ('scripts', 'bayesfilter'):
        if name in sys.modules:
            module = sys.modules[name]
            paths = list(getattr(module, '__path__', ()))
            if not paths or any(not Path(p).resolve().is_relative_to(root) for p in paths):
                raise ValueError('load recovery in a fresh process; shared source already imported')
    os.environ['BAYESFILTER_DIAGNOSTIC_CONSUMER_ROOT'] = str(consumer)
    os.environ['BAYESFILTER_DIAGNOSTIC_GPU_EVIDENCE'] = str(campaign/'preflight/evidence')
    os.environ['BAYESFILTER_OP_LIB_DIR'] = str(root/'bayesfilter/ops')
    sys.path[:0] = [str(root), str(root/'scripts'), str(consumer)]
    snapshot = importlib.import_module('scripts.modern_hmc_source_snapshot')
    snapshot.validate_snapshot(source, expected_root=root)
    frozen = importlib.import_module('scripts.run_modern_hmc_macrofinance_frozen_diagnostic')
    return SimpleNamespace(source=source, root=root, snapshot=snapshot, frozen=frozen)


def worker(args):
    if (os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true'
            or not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')):
        raise ValueError('GPU UUID and memory growth required before framework import')
    out = args.output/'evidence'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = dict(status='running', plan=args.plan, posterior_qualification=False,
        external_consumer_admitted=False, default_promoted=False, no_ranking=True)
    manifest = dict(plan=args.plan, command=[sys.executable, *sys.argv],
        started_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=args.wall_seconds,
        recovery_entrypoint=str(Path(__file__).resolve()), recovery_entrypoint_sha256=digest(__file__),
        environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES',
            'TF_FORCE_GPU_ALLOW_GROWTH','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS')},
        gpu_trust_basis='trusted_escalated_gpu_execution', source=str(args.source))
    write(out/'manifest.json', manifest)

    def mark(stage, details=None):
        write(out/'progress.json', dict(stage=stage, details=details, elapsed_seconds=time.monotonic()-started))

    try:
        runtime = load_runtime(args.campaign)
        frozen = runtime.frozen
        paths, _ = frozen.source_paths(args.campaign)
        before = {str(path):digest(path) for path in paths}
        cohort = read(args.campaign/'frozen-cohort.json')
        from scripts.resume_modern_hmc_macrofinance_public import copy_parent, resume_checkpoint
        mark('copy_and_validate_frozen_checkpoint')
        parent = copy_parent(args.source, out, frozen.INPUTS)
        manifest.update(source_sha256=before, inputs_sha256=digest(frozen.INPUTS),
            git_commit=runtime.source['git_commit'],
            data_version=parent.get('data_version', 'saved 48-period MacroFinance synthetic fixture'),
            seed_namespaces=dict(qualification=cohort['qualification_seed'], reload=cohort['reload_seed']))
        from bayesfilter.runtime.gpu_provenance import selected_nvidia_gpu, query_nvidia_smi_gpus
        manifest['physical_gpu'] = selected_nvidia_gpu(query_nvidia_smi_gpus())
        write(out/'manifest.json', manifest)
        from scripts.diagnose_modern_hmc_macrofinance_public import build
        mark('build_unchanged_adapter')
        adapter, memory = build(read(frozen.INPUTS), device='GPU', gpu_evidence=frozen.GPU_EVIDENCE)
        import tensorflow as tf
        import tensorflow_probability as tfp
        manifest.update(memory_policy=memory, tensorflow=tf.__version__, tensorflow_probability=tfp.__version__,
            dtype='float64', tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(), use_xla=True,
            loaded_local_modules=runtime.snapshot.audit_imports(runtime.source))
        write(out/'manifest.json', manifest)
        from bayesfilter.runtime.execution_budget import execution_budget, execution_progress
        deadline = started+args.wall_seconds-900.0
        mark('validate_and_resume_checkpoint')
        with execution_budget(deadline=deadline, check=lambda: time.monotonic() < deadline,
                              replace_wall_time_limit=True), execution_progress(mark):
            report['execution'] = resume_checkpoint(out/'tuning/tuning_checkpoint.json', adapter, out)
        write(out/'controller-outcome.json', report['execution'])
        if report['execution']['completion_status'] == 'complete':
            from scripts.continue_modern_hmc_macrofinance_broad import reload
            (out/'reload').mkdir()
            with execution_budget(deadline=started+args.wall_seconds), execution_progress(mark):
                report['reload'] = reload(adapter, out, out/'reload', mark, seed_root=cohort['reload_seed'])
        report['parent_evidence_unchanged'] = all(digest(p)==sha
            for p,sha in read(out/'parent-evidence.json').items())
        report['sources_unchanged'] = all(digest(p)==sha for p,sha in before.items())
        report['loaded_local_modules'] = runtime.snapshot.audit_imports(runtime.source)
        if not report['parent_evidence_unchanged'] or not report['sources_unchanged']:
            raise ValueError('original evidence or source changed during recovery')
        report['allocator_bytes'] = tf.config.experimental.get_memory_info('GPU:0')
        from scripts.modern_hmc_campaign_status import completion_status
        report['status'] = completion_status(report['execution']['completion_status'])
    except Exception as exc:
        report.update(status='error', error=repr(exc), traceback=traceback.format_exc())
    report.update(elapsed_seconds=time.monotonic()-started, finished_utc=datetime.now(timezone.utc).isoformat())
    write(out/'result.json', report)
    return 0 if report['status']=='completed' else 1


def supervise(args):
    runtime = load_runtime(args.campaign)
    from scripts.run_modern_hmc_filter_health_gpu import Reassessment, checkpoint_progress, stop_worker, reload_gpu_uuid
    from bayesfilter.runtime.gpu_provenance import query_nvidia_smi_gpus
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    receipt = dict(status='selecting_device', active_run=True, plan=args.plan,
        command=[sys.executable, *sys.argv], started_utc=datetime.now(timezone.utc).isoformat(),
        max_wall_seconds=args.wall_seconds, observations=[],
        recovery_entrypoint_sha256=digest(__file__), frozen_source=str(args.campaign/'isolated-source.json'))
    write(args.output/'manifest.json', receipt)
    child, monitor = None, Reassessment()

    def interrupted(signum, frame):
        raise InterruptedError('supervisor signal '+str(signum))

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        required = reload_gpu_uuid(args.source)
        while True:
            now = time.monotonic()
            if now-started >= args.wall_seconds-1020:
                raise TimeoutError('allocation exhausted waiting for original GPU capacity')
            if monitor.due(now, waiting=True):
                observation = monitor.observe(now, query_nvidia_smi_gpus(), required_uuid=required)
                receipt['observations'].append(observation)
                receipt['heartbeat_utc'] = datetime.now(timezone.utc).isoformat()
                write(args.output/'manifest.json', receipt)
                selected = observation['selected_for_next_worker']
                if selected is not None:
                    break
            time.sleep(1)
        command = [sys.executable, str(Path(__file__).resolve()), '--campaign', str(args.campaign),
            '--source', str(args.source), '--output', str(args.output), '--plan', args.plan,
            '--wall-seconds', str(args.wall_seconds-(time.monotonic()-started)-120), '--worker']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=required, TF_FORCE_GPU_ALLOW_GROWTH='true',
            PYTHONPATH=str(runtime.root), PYTHONDONTWRITEBYTECODE='1', TF_CPP_MIN_LOG_LEVEL='2',
            TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='2', OMP_NUM_THREADS='2')
        receipt.update(status='running', worker_command=command, selected_gpu=selected,
                       capacity_wait_seconds=time.monotonic()-started)
        with (args.output/'worker.log').open('x') as log:
            child = subprocess.Popen(command, cwd=runtime.root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            receipt['worker_pid'] = child.pid
            write(args.output/'manifest.json', receipt)
            last = 0.0
            while child.poll() is None:
                now = time.monotonic()
                if now-started >= args.wall_seconds-5:
                    receipt['status']='wall_budget_exhausted'
                    stop_worker(child)
                    break
                if now-last >= 60:
                    receipt.update(heartbeat_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=now-started, numerical_progress=checkpoint_progress(args.output))
                    write(args.output/'manifest.json', receipt)
                    last=now
                if monitor.due(now):
                    receipt['observations'].append(monitor.observe(now, query_nvidia_smi_gpus(),
                        active_uuid=required, required_uuid=required))
                    write(args.output/'manifest.json', receipt)
                time.sleep(1)
        receipt['exit_code']=child.returncode
        if receipt['status']=='running':
            result=read(args.output/'evidence/result.json')
            receipt.update(status=result['status'], execution=result.get('execution'))
            if result.get('error'):
                receipt.update(error=result['error'], traceback=result.get('traceback'))
    except Exception as exc:
        receipt.update(status='error', error=repr(exc), traceback=traceback.format_exc())
    finally:
        stop_worker(child)
        receipt.update(active_run=False, elapsed_seconds=time.monotonic()-started,
                       finished_utc=datetime.now(timezone.utc).isoformat())
        write(args.output/'manifest.json', receipt)
    return 0 if receipt['status']=='completed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','source','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--wall-seconds', type=float, required=True)
    parser.add_argument('--worker', action='store_true')
    args=parser.parse_args()
    for name in ('campaign','source','output'):
        setattr(args,name,getattr(args,name).resolve())
    if not math.isfinite(args.wall_seconds) or args.wall_seconds<=1020:
        parser.error('finite allowance must exceed export and closeout reserves')
    return worker(args) if args.worker else supervise(args)


if __name__=='__main__':
    raise SystemExit(main())
