"""Resume preserved MacroFinance diagnostic trials; no posterior admission."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = 'docs/plans/bayesfilter-modern-hmc-public-gpu-recovery-2026-10-24.md'
from scripts.modern_hmc_campaign_status import completion_status


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + chr(10))
    temporary.replace(path)


def retry_allowance(events, missing):
    """Fund only the next retry of a charged trial without a saved result."""
    charges = {}
    for event in events:
        if event.get('event') != 'numerical_chunk_charged':
            continue
        key = (event['work_item_id'], event['trial_ordinal'], event['trial_chunk_index'])
        cost = event['gradient_work']
        if type(cost) is not int or cost <= 0:
            raise ValueError('invalid attempted-work charge')
        signature = (cost, tuple(event['seed']), event['transitions'])
        if key in charges and charges[key] != signature:
            raise ValueError('inconsistent charges for interrupted trial')
        charges[key] = signature
    if not set(missing).issubset(charges):
        raise ValueError('unrecorded retry requested')
    return sum(charges[key][0] for key in missing)


def copy_parent(source, output, inputs):
    """Preserve the old attempt and check its runtime and numerical sources."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError('recovery must be outside its parent evidence')
    manifest = read(source/'manifest.json')
    supervisor = read(source.parent/'manifest.json')
    if supervisor.get('active_run') or supervisor.get('status') not in ('wall_budget_exhausted', 'resource_stopped'):
        raise ValueError('terminal wall-budget parent required')
    if digest(inputs) != manifest['inputs_sha256']:
        raise ValueError('recovery inputs differ from frozen target')
    for name, expected in manifest['source_sha256'].items():
        if digest(name) != expected:
            raise ValueError('frozen numerical source changed: '+name)
    spec = read(source/'tuning/execution_spec.json')['execution']
    runtime = spec['runtime_policy']
    if (runtime['device_type'] != 'GPU' or not spec['config']['use_xla']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != runtime['cuda_visible_devices']
            or manifest['physical_gpu']['uuid'] != runtime['cuda_visible_devices']):
        raise ValueError('recovery must preserve GPU/XLA runtime and UUID')
    preserved = {str(p): digest(p) for p in source.rglob('*') if p.is_file()}
    shutil.copytree(source/'tuning', output/'tuning')
    shutil.copyfile(source/'design.json', output/'design.json')
    write(output/'parent-evidence.json', preserved)
    return manifest


def resume_checkpoint(checkpoint, adapter, output):
    """Resume the numerical controller with an explicit lost-call allowance."""
    from bayesfilter.inference.hmc_candidate_set_checkpoint import load_numerical_tuning_checkpoint
    from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
    from bayesfilter.inference.hmc_candidate_set_retained import export_hmc_candidate_retained_runners
    from scripts.resume_modern_hmc_full_model_qualification import inherited_attempts
    from scripts.resume_modern_hmc_bridge import audit_recovery

    before = read(checkpoint)
    events, missing = inherited_attempts(checkpoint)
    extra = retry_allowance(events, missing)
    binding, controller = load_numerical_tuning_checkpoint(checkpoint, adapter=adapter)
    old_config = controller.config.payload()
    old_cap = controller.config.max_gradient_work
    if old_cap is None and extra:
        raise ValueError('this recovery expects a finite recorded gradient-work cap')
    if extra:
        controller.config = replace(controller.config, max_gradient_work=old_cap+extra)
        controller._accounting.append(dict(event='infrastructure_retry_allowance',
            previous_max_gradient_work=old_cap, max_gradient_work=old_cap+extra,
            additional_gradient_work=extra, missing_trial_keys=sorted(missing),
            reason='charged native calls without durable results; no completed-trial replay'))
    # Compare serialized checkpoint values consistently (tuples become lists).
    expected_config = json.loads(json.dumps(old_config))
    if extra:
        expected_config['max_gradient_work'] = old_cap+extra
    write(Path(output)/'recovery-before.json', dict(binding_hash=binding.binding_hash,
        original_config=old_config, expected_config=expected_config,
        additional_gradient_work=extra, inherited_missing_keys=sorted(missing)))
    run = run_typed_hmc_candidate_set(binding.typed_adapter, controller.config,
        output_dir=Path(checkpoint).parent, _resume_controller=controller)
    after = json.loads(json.dumps(run.result.payload()))
    if (binding.binding_hash != before['binding_hash'] or after['config'] != expected_config
            or after['candidates'][:len(before['result']['candidates'])] != before['result']['candidates']
            or after['observations'][:len(before['result']['observations'])] != before['result']['observations']):
        raise ValueError('recovery changed frozen identity, candidates or observations')
    audit = audit_recovery(run.result, binding, inherited_events=events, inherited_missing_keys=missing)
    write(Path(output)/'native-audit.json', audit)
    paths = export_hmc_candidate_retained_runners(candidate_set_result=run.result,
        retained_binding=binding, output_dir=Path(output)/'members')
    if set(paths) != set(run.result.verified_candidate_ids):
        raise ValueError('recovery export differs from verified set')
    write(Path(output)/'members.json', {cid: str(path) for cid,path in paths.items()})
    return dict(completion_status=run.result.completion_status, final_status=run.result.final_status,
        candidate_states=dict(run.result.candidate_states), verified=sorted(paths),
        total_candidates=len(run.result.candidates), native_audit=audit,
        all_verified_members_exported=True, additional_gradient_work=extra, no_ranking=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=float,required=True)
    parser.add_argument('--plan',default=PLAN)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(status='running',mode='resume',plan=args.plan,posterior_qualification=False,
        default_promoted=False,evidence_role='actual_consumer_gpu_diagnostic_composition')
    manifest=dict(command=[sys.executable,*sys.argv],plan=args.plan,started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inputs_sha256=digest(args.inputs),environment={key:os.environ.get(key) for key in
        ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS')},
        harness_sha256={str(p):digest(p) for p in (Path(__file__).resolve(),
            ROOT/'scripts/resume_modern_hmc_bridge.py',ROOT/'scripts/resume_modern_hmc_full_model_qualification.py',
            ROOT/'scripts/modern_hmc_campaign_status.py')},
        source=str(args.source.resolve()),wall_seconds=args.wall_seconds)
    write(args.output/'manifest.json',manifest)
    def mark(stage,details=None):
        write(args.output/'progress.json',dict(stage=stage,details=details,elapsed_seconds=time.monotonic()-started))
    try:
        if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true':
            raise ValueError('memory growth required before TensorFlow import')
        parent=copy_parent(args.source,args.output,args.inputs)
        manifest['source_sha256']=parent['source_sha256']
        from bayesfilter.runtime.gpu_provenance import selected_nvidia_gpu,query_nvidia_smi_gpus
        manifest['physical_gpu']=selected_nvidia_gpu(query_nvidia_smi_gpus())
        manifest['gpu_trust_basis']='trusted_escalated_gpu_execution'
        write(args.output/'manifest.json',manifest)
        sys.path.insert(0,str(ROOT/'scripts'))
        from diagnose_modern_hmc_macrofinance_public import build,CONSUMER
        sys.path.insert(0,str(CONSUMER))
        mark('build_unchanged_adapter')
        adapter,memory=build(read(args.inputs),device='GPU')
        import tensorflow as tf
        import tensorflow_probability as tfp
        manifest.update(memory_policy=memory,tensorflow=tf.__version__,tensorflow_probability=tfp.__version__,
            tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),dtype='float64')
        write(args.output/'manifest.json',manifest)
        from bayesfilter.runtime.execution_budget import execution_budget,execution_progress
        mark('validate_and_resume_checkpoint')
        deadline=started+args.wall_seconds
        with execution_budget(check=lambda:time.monotonic()<deadline,replace_wall_time_limit=True),execution_progress(mark):
            report['execution']=resume_checkpoint(args.output/'tuning/tuning_checkpoint.json',adapter,args.output)
        report['parent_evidence_unchanged']=all(digest(p)==h for p,h in read(args.output/'parent-evidence.json').items())
        report['sources_unchanged']=all(digest(p)==h for p,h in manifest['source_sha256'].items())
        if not report['parent_evidence_unchanged'] or not report['sources_unchanged']:
            raise ValueError('original evidence or source changed during recovery')
        report['allocator_bytes']=tf.config.experimental.get_memory_info('GPU:0')
        report['status']=completion_status(report['execution']['completion_status'])
    except Exception as exc:
        report.update(status='error',error=repr(exc),traceback=traceback.format_exc())
    report.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output/'result.json',report)
    print(json.dumps({k:report[k] for k in ('status','elapsed_seconds')}),flush=True)
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
