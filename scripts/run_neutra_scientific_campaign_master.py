#!/usr/bin/env python3
"""Bounded scientific NeuTra campaign for simple and randomized mixtures.

The master is orchestration only. TensorFlow is imported inside workers after
the memory-growth environment is established. Every phase refreshes the
versioned state/matrix/next-phase/pricing files and preserves failed attempts.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

LIVE = Path("/home/ubuntu/python/BayesFilter")
ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/bayesfilter-neutra-source-fit-execution-2026-10-05.md"
DEVELOPMENT_PLAN = "docs/plans/bayesfilter-neutra-development-hmc-2026-10-05.md"
ATTRIBUTION_PLAN = "docs/plans/bayesfilter-neutra-nonlinearity-attribution-plan-2026-10-05.md"
FORWARD_REVERSE_PLAN = "docs/plans/bayesfilter-neutra-naf-forward-reverse-master-2026-10-06.md"
ROUTE_LEDGER = "docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/c0/route_ledger.json"
CAMPAIGN = LIVE / "docs/plans/artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1"
PREVIOUS = (LIVE / "docs/plans/artifacts/neutra-scientific-2026-10-04/campaign-r1",
            LIVE / "docs/plans/artifacts/neutra-scientific-2026-10-04/campaign-r2",
            LIVE / "docs/plans/artifacts/neutra-scientific-2026-10-04/campaign-r3")
SHARED = LIVE / "docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1"
sys.path.insert(0, str(ROOT))
from bayesfilter.testing.neutra_scientific_design import (
    METHODS, CALIBRATION_TARGETS as DEV_TARGETS, FINAL_TARGETS, FIT_SEEDS,
    target_catalog, profile_candidates, student_candidates, combine_profile,
)

# Cumulative allocation including the owner's additional 24 GPU-process and
# 24 CPU-core hours on each of October 5 and October 6. Prior costs stay charged.
LIMITS = {"gpu_process_seconds": 178800.0, "cpu_core_seconds": 184800.0}
TEST_PATHS = (
    "tests/test_neutra_six_method_controls.py",
    "tests/test_neutra_warm_start_pipeline.py",
    "tests/test_neutra_six_method_master.py",
    "tests/test_neutra_scientific_campaign.py",
    "tests/test_neutra_scientific_master.py",
    "tests/test_neutra_source_fit_remedy.py",
    "tests/test_neutra_single_authority.py",
    "tests/test_neutra_source_fit_validation.py",
    "tests/test_neutra_hmc_route_policy.py",
    "tests/test_neutra_nonlinearity_attribution.py",
    "tests/test_neutra_attribution_controller.py",
    "tests/test_neutra_forward_reverse.py",
    "tests/test_neutra_forward_reverse_controller.py",
    "tests/test_neutra_forward_reverse_criteria.py",
)
AUTHOR_FIXTURE = "docs/plans/artifacts/neutra-warm-start-master-2026-09-29/author-reference-r2.json"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_snapshot():
    files = list((LIVE / "bayesfilter").rglob("*.py"))
    ledger=read(LIVE/ROUTE_LEDGER)
    files.extend(LIVE/row['path'] for row in (*ledger['routes'],*ledger['discovery']['exclusions'])
                 if (LIVE/row['path']).is_file())
    files.extend(
        LIVE / name
        for name in (
            "scripts/run_neutra_scientific_campaign_master.py",
            "scripts/run_neutra_scientific_campaign.sh",
            "scripts/neutra_attribution_campaign.py",
            "scripts/neutra_forward_reverse_campaign.py",
            "scripts/supervise_neutra_attribution.py",
            "scripts/run_neutra_six_method_master.py",
            "scripts/run_neutra_six_method_campaign.sh",
            *TEST_PATHS,
            "tests/conftest.py",
            "pytest.ini",
            AUTHOR_FIXTURE,
            PLAN,
            DEVELOPMENT_PLAN,
            ATTRIBUTION_PLAN,
            FORWARD_REVERSE_PLAN,
            "docs/plans/bayesfilter-neutra-forward-warm-start-criterion-repair-2026-10-06.md",
            ROUTE_LEDGER,
            "docs/plans/bayesfilter-neutra-source-and-fit-remedy-plan-2026-10-04.md",
            "docs/plans/bayesfilter-neutra-representation-check-2026-10-05.md",
            "docs/plans/bayesfilter-neutra-generic-recovery-and-transfer-plan-2026-10-03.md",
        )
    )
    hashes = {
        str(path.relative_to(LIVE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = CAMPAIGN / ("source-" + digest[:16])
    if not destination.exists():
        destination.mkdir(parents=True)
        for name in hashes:
            copied = destination / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(LIVE / name, copied)
        write(
            destination / "source.json",
            {
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=LIVE, text=True
                ).strip(),
                "sha256": hashes,
                "snapshot_id": digest,
                "dirty_worktree_preserved": True,
            },
        )
    for name, expected in hashes.items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"scientific source snapshot differs: {name}")
    return destination


def conservative_shared_remaining():
    config, state = read(SHARED / "config.json"), read(SHARED / "state.json")
    used = {key: sum(row.get(key, 0.0) for row in state["attempts"]) for key in LIMITS}
    for row in state["attempts"]:
        if row.get("cpu_accounting_estimated"):
            used["cpu_core_seconds"] += max(
                0.0,
                row.get("cpu_core_seconds_hard_upper_bound", config["cpu_core_seconds"])
                - row.get("cpu_core_seconds", 0.0),
            )
    return {key: config[key] - used[key] for key in LIMITS}


def worker(specification):
    spec = read(specification)
    output = Path(spec["output"])
    started = time.monotonic()
    cpu_teacher = spec.get('phase') == 'forward_reverse_teacher'
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1' if cpu_teacher else str(spec["gpu"])
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
        raise RuntimeError("TF_FORCE_GPU_ALLOW_GROWTH must be set before TensorFlow import")
    resource.setrlimit(resource.RLIMIT_CPU, (int(spec["cpu_limit"]), int(spec["cpu_limit"])))
    sys.path.insert(0, str(ROOT))
    manifest = {
        "command": [sys.executable, str(Path(__file__).resolve()), "worker", "--spec", str(specification)],
        "environment": sys.executable,
        "plan": plan_for_phase(spec.get('phase')),
        "source": str(ROOT / "source.json"),
        "git_commit": read(ROOT / "source.json")["git_commit"],
        "seed": spec.get("seed"),
        "method": spec.get("method"),
        "target": spec.get("target"),
        "profile_id": spec.get("profile", {}).get("profile_id"),
        "role": spec.get("role"),
        "gpu": spec["gpu"],
        "TF_FORCE_GPU_ALLOW_GROWTH": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
        "dtype": "float64_reference",
        "jit_compile": True,
        "tf32": True,
        "tf32_applies_to_float64": False,
        "profile": spec.get("profile"),
        "data_version": "synthetic_target_sha256",
        "result_file": str(output / "result.json"),
        "output": str(output),
        "trust_basis": "trusted_fixed_scientific_campaign_wrapper",
        "status": "running",
        "started_unix": time.time(),
    }
    write(output / "manifest.json", manifest)
    code = 1
    try:
        import tensorflow as tf

        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

        if cpu_teacher:
            if tf.config.list_physical_devices('GPU'):
                raise RuntimeError('CPU teacher unexpectedly sees GPU devices')
            manifest['gpu_devices_intentionally_hidden'] = True
            manifest['memory_policy'] = {'mode':'cpu_only_gpu_intentionally_hidden'}
        else:
            manifest["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(True)
        write(output / "manifest.json", manifest)
        from bayesfilter.testing.neutra_scientific_campaign import run_trial, serializable

        if spec.get("action") == "preflight":
            @tf.function(input_signature=[tf.TensorSpec([32, 2], tf.float64)],
                         jit_compile=True, autograph=False)
            def kernel(x):
                return tf.reduce_sum(tf.square(x), axis=1)

            values = kernel(tf.ones([32, 2], tf.float64))
            tf.debugging.assert_equal(values, tf.fill([32], tf.constant(2.0, tf.float64)))
            if "GPU" not in values.device:
                raise RuntimeError("scientific preflight executed off GPU")
            result = {"status": "passed", "device": values.device,
                      "traces": kernel.experimental_get_tracing_count(),
                      "compiled_kernel": "squared_norm", "scientific_promotion": False}
        elif spec.get('phase') == 'matched_fit':
            from bayesfilter.testing.neutra_source_fit_remedy import run_matched_fit
            result = run_matched_fit(spec['specification'],spec['profile'],spec['seed'],output)
        elif spec.get('phase') == 'optimizer_repair':
            from bayesfilter.testing.neutra_source_fit_remedy import run_optimizer_repair
            result = run_optimizer_repair(spec['specification'],spec['profile'],spec['seed'],output)
        elif spec.get('phase') == 'representation':
            from bayesfilter.testing.neutra_source_fit_remedy import run_representation
            result = run_representation(spec['specification'],spec['profile'],spec['seed'],output)
        elif spec.get('phase') == 'representation_debug':
            from bayesfilter.testing.neutra_source_fit_remedy import diagnose_representation_failure
            result = diagnose_representation_failure(spec['specification'],spec['profile'],spec['seed'],output)
        elif spec.get('phase') == 'development_hmc':
            from bayesfilter.testing.neutra_source_fit_validation import validate_development_map
            result = validate_development_map(spec['specification'],spec['profile'],spec['seed'],output)
        elif spec.get('phase') == 'nonlinearity_attribution':
            from bayesfilter.testing.neutra_nonlinearity_attribution import run_attribution
            result = run_attribution(spec['specification'],spec['profile'],spec['seed'],output)
        elif spec.get('phase') in ('forward_reverse_teacher','forward_reverse_fit'):
            from bayesfilter.testing.neutra_forward_reverse import prepare_teacher, run_fit
            if cpu_teacher:
                result = prepare_teacher(spec['specification'],spec['profile'],spec['seed'],output,
                    method=spec['profile']['method'])
            else:
                result = run_fit(spec['specification'],spec['profile'],spec['seed'],output)
        else:
            result = run_trial(
                spec["method"], spec["target"], spec["specification"], spec["profile"],
                spec["seed"], output, role=spec["role"], exact_teacher=spec.get("exact_teacher", False),
            )
        write(output / "result.json", serializable(result))
        manifest.update(target_signature=result.get("target_signature"), tensorflow_version=tf.__version__,
                        status="complete", allocator=None if cpu_teacher else tf.config.experimental.get_memory_info("GPU:0"))
        code = 0
    except Exception as error:
        traceback.print_exc()
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        manifest.update(
            wall_seconds=time.monotonic() - started,
            cpu_core_seconds=usage.ru_utime + usage.ru_stime,
            finished_unix=time.time(),
            gpu_process_seconds=0. if cpu_teacher else time.monotonic() - started,
            execution_cwd=str(ROOT),
        )
        write(output / "manifest.json", manifest)
    return code


class CampaignStop(RuntimeError):
    """A budget, shared-harness or artifact veto. Candidate screens are results."""


def plan_for_phase(phase):
    if phase in ('forward_reverse_teacher','forward_reverse_fit'):return FORWARD_REVERSE_PLAN
    if phase=='nonlinearity_attribution':return ATTRIBUTION_PLAN
    if phase=='development_hmc':return DEVELOPMENT_PLAN
    return PLAN


def jobs():
    yield 'check', {'phase': 'preflight'}
    yield 'preflight', {'phase': 'preflight'}
    for target in DEV_TARGETS:
        for profile in student_candidates():
            yield f'student-{target}-{profile["profile_id"]}', dict(phase='student_calibration',method='exact_teacher',target=target)
    for method in METHODS:
        for target in DEV_TARGETS:
            for profile in profile_candidates(method):
                yield f'calibrate-{method}-{target}-{profile["profile_id"]}', dict(phase='calibration',method=method,target=target)
    for target in FINAL_TARGETS:
        yield f'exact-teacher-{target}', dict(phase='exact_teacher',method='exact_teacher',target=target)
    for method in METHODS:
        for target in FINAL_TARGETS:
            for seed in FIT_SEEDS:
                yield f'final-{method}-{target}-s{seed}', dict(phase='final',method=method,target=target,seed=seed)


def audit_campaign(root, shared):
    state = read(root/'state.json')
    latest = {r['job']: r for r in state['attempts']}
    errors, cells = [], []
    sources = {r['source'] for r in state['attempts']}
    for name in sources:
        try:
            source = read(Path(name)/'source.json')
            for file, digest in source['sha256'].items():
                if hashlib.sha256((Path(name)/file).read_bytes()).hexdigest() != digest:
                    errors.append('source differs: '+file)
        except (OSError, ValueError, KeyError) as error:
            errors.append('source unreadable: '+str(error))
    for r in state['attempts']:
        charges = [v for v in shared['attempts'] if v.get('output') == r['output']]
        if len(charges) != 1:
            errors.append('charge count differs: '+r['job'])
        elif any(abs(charges[0].get(k,0)-r.get(k,0)) > 1e-8 for k in LIMITS):
            errors.append('charge amount differs: '+r['job'])
    for job, fields in jobs():
        row = latest.get(job)
        cell = {'job':job, **fields, 'status':'not_run'}
        cells.append(cell)
        if row is None:
            continue
        output = Path(row['output'])
        cell['output'] = str(output)
        if row['status'] != 'complete':
            cell['status'] = 'infrastructure_failed'
            continue
        try:
            result = read(output/'result.json')
            manifest = read(output/'manifest.json')
            cell['status'] = result['status']
            if result.get('scientific_promotion') is not False:
                errors.append(job+': invalid promotion flag')
            if manifest['source'] != row['source']+'/source.json':
                errors.append(job+': manifest source differs')
            if row['device'] == 'gpu':
                policy = manifest.get('memory_policy',{})
                if not policy.get('all_physical_devices_memory_growth') or not policy.get('configured_before_logical_device_initialization'):
                    errors.append(job+': memory policy absent/invalid')
                if manifest.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true':
                    errors.append(job+': growth environment missing')
            if row.get('phase') not in ('preflight',) and row['device'] != 'none':
                spec = read(output/'spec.json')
                target = read(output/'target.json')
                if target != spec['specification'] or target != read(root/'target-catalog.json')[row['target']]:
                    errors.append(job+': target changed')
                digest = hashlib.sha256(json.dumps(target,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                if result.get('target_signature') != digest or manifest.get('target_signature') != digest:
                    errors.append(job+': target signature mismatch')
                if result.get('profile') != spec['profile'] or result.get('seed') != spec['seed']:
                    errors.append(job+': seed/profile mismatch')
                for file, expected in manifest.get('artifact_sha256',{}).items():
                    if hashlib.sha256((output/file).read_bytes()).hexdigest() != expected:
                        errors.append(job+': artifact differs: '+file)
                if result['status'] in ('passed','map_failed'):
                    for label in ('forward','student'):
                        probe = read(output/(label+'-post-training-1000.json'))
                        checkpoint = read(output/(label+'-checkpoint.json'))
                        frozen = read(output/(label+'-frozen.json'))
                        checkpoint_hash = checkpoint.get('state_hash',checkpoint.get('checkpoint_hash'))
                        body = {k:v for k,v in checkpoint.items() if k not in ('state_hash','checkpoint_hash')}
                        calculated = hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                        if calculated != checkpoint_hash or frozen['training_state_hash'] != checkpoint_hash:
                            errors.append(job+': checkpoint mismatch '+label)
                        body = {k:v for k,v in frozen.items() if k != 'transport_hash'}
                        calculated = hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                        if calculated != frozen['transport_hash'] or frozen['target_signature'] != digest:
                            errors.append(job+': frozen map mismatch '+label)
                        if probe.get('rows') != 1000 or not probe.get('complete'):
                            errors.append(job+': incomplete probe '+label)
                        stage = result['forward' if label == 'forward' else 'final']
                        if not stage.get('checkpoint_reloaded') or stage.get('transport_hash') != frozen['transport_hash']:
                            errors.append(job+': checkpoint not assessed '+label)
                    for label in ('forward_training','rkl_training'):
                        training = result[label]
                        if training['updates'] != training['requested_updates'] or training['batch_size'] <= 1 or training['samplewise_loop']:
                            errors.append(job+': incomplete/bad training '+label)
                        if not training['jit_compile'] or 'GPU' not in training['training_device']:
                            errors.append(job+': wrong training device/compilation '+label)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(job+': missing/invalid artifact '+str(error))
    missing = [c['job'] for c in cells if c['status'] in ('not_run','infrastructure_failed')]
    return {'status':'failed' if errors else 'passed', 'complete':not missing, 'errors':errors,
            'missing_or_failed_jobs':missing, 'cells':cells, 'scientific_promotion':False}


class Controller:
    def __init__(self):
        CAMPAIGN.mkdir(parents=True, exist_ok=True)
        self.state = read(CAMPAIGN/'state.json') if (CAMPAIGN/'state.json').exists() else {
            'schema':'bayesfilter.neutra.scientific_campaign.v2', 'status':'prepared',
            'attempts':[], 'active':None, 'limits':LIMITS, 'plan':PLAN, 'source':None,
            'created_unix':time.time()}
        self.catalog = target_catalog()
        path = CAMPAIGN/'target-catalog.json'
        if path.exists() and read(path) != self.catalog:
            raise CampaignStop('frozen target catalogue differs')
        if not path.exists():write(path,self.catalog)
        self.source = Path(self.state['source']) if self.state.get('source') else None
        self.source_checked = False
        self.recover()

    def ensure_source(self):
        if not self.source_checked:
            self.source = source_snapshot()
            self.source_checked = True
            self.state['source'] = str(self.source)
            write(CAMPAIGN/'state.json',self.state)
        return self.source

    def used(self):
        attempts = list(self.state['attempts'])
        for path in PREVIOUS:
            attempts.extend(read(path/'state.json')['attempts'])
        return {k:sum(r.get(k,0.) for r in attempts) for k in LIMITS}

    def remaining(self):
        shared, used = conservative_shared_remaining(), self.used()
        return {k:min(shared[k],LIMITS[k]-used[k]) for k in LIMITS}

    def settle_pending_charges(self):
        """Caller holds the shared master lock; replay-safe across partial writes."""
        if self.state.get('active'):
            raise CampaignStop('cannot settle charges during active worker')
        for path in sorted(CAMPAIGN.glob('*/pending-charge.json')):
            row=read(path)
            if row.get('accounting_status')=='settled':
                continue
            if Path(row['output']).resolve()!=path.parent.resolve():
                raise CampaignStop('pending charge output mismatch')
            for key in (*LIMITS,'wall_seconds'):
                if not math.isfinite(row[key]) or row[key]<0:
                    raise CampaignStop('invalid pending charge')
            row={**row,'source':str(path.parent),'attempt':1,
                 'exit_code':row.get('exit_code',0 if row['status']=='complete' else 1),
                 'accounting_status':'settled'}
            if not any(r['output']==row['output'] for r in self.state['attempts']):
                self.state['attempts'].append(row)
            write(CAMPAIGN/'state.json',self.state)
            shared=read(SHARED/'state.json')
            if not any(r.get('output')==row['output'] for r in shared['attempts']):
                shared['attempts'].append(row)
                write(SHARED/'state.json',shared)
            write(path,{**row,'settled_unix':time.time()})

    def sync(self, status, next_action):
        self.state.update(status=status,next_action=next_action,remaining=self.remaining(),updated_unix=time.time())
        write(CAMPAIGN/'state.json',self.state)
        write(CAMPAIGN/'next-phase.json',{'phase':status,'next_action':next_action,'remaining':self.remaining(),
            'resume_command':'bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_scientific_campaign.sh resume'})
        latest = {r['job']:r for r in self.state['attempts']}
        rows = []
        for job,fields in jobs():
            r = latest.get(job)
            path = Path(r['output'])/'result.json' if r else None
            result = read(path) if path and path.exists() else {}
            rows.append({'job':job,**fields,'status':result.get('status',r['status'] if r else 'not_run'),
                         'result':str(path) if path else None})
        write(CAMPAIGN/'matrix.json',{'rows':rows,'scientific_promotion':False})
        self.write_pricing()

    def write_pricing(self):
        costs = {}
        for phase in ('student_calibration','calibration','exact_teacher','final'):
            rows = [r for r in self.state['attempts'] if r['phase']==phase and r['device']=='gpu']
            costs[phase] = {'measured_workers':len(rows),'mean_wall_seconds':sum(r['wall_seconds'] for r in rows)/len(rows) if rows else None}
        write(CAMPAIGN/'pricing.json',{'limits_cumulative_all_revisions':LIMITS,'used_cumulative':self.used(),
            'remaining':self.remaining(),'measured_costs':costs,'planned_decisions':len(list(jobs())),
            'forecast_status':'partial_measurements_conditional_on_calibration','scientific_promotion':False})

    def recover(self):
        row = self.state.get('active')
        if row is None:return
        try:
            process = Path(f'/proc/{row["pid"]}/status').read_text()
            if '\nState:\tZ' not in process:raise CampaignStop('worker still active')
        except (ProcessLookupError,FileNotFoundError):pass
        path = Path(row['output'])/'manifest.json'
        manifest = read(path) if path.exists() else {}
        self.finish(row,0 if manifest.get('status')=='complete' else 125,manifest)

    def finish(self,row,code,manifest):
        wall = manifest.get('wall_seconds',min(time.time()-row['started_unix'],row['wall_limit']))
        cpu = manifest.get('cpu_core_seconds',row['cpu_limit'])
        completed = {**row,'exit_code':code,'status':'complete' if code==0 else 'failed',
                     'wall_seconds':wall,'gpu_process_seconds':wall if row['device']=='gpu' else 0.,'cpu_core_seconds':cpu}
        if not any(r['output']==row['output'] for r in self.state['attempts']):
            self.state['attempts'].append(completed)
        self.state['active'] = None
        shared = read(SHARED/'state.json')
        if not any(r.get('output')==row['output'] for r in shared['attempts']):
            shared['attempts'].append({'job':'neutra-source-fit-20261005-'+row['job']+'-r'+str(row['attempt']),
                'phase':'neutra_scientific_simple_mixtures','output':row['output'],'status':completed['status'],
                'wall_seconds':wall,**{k:completed[k] for k in LIMITS}})
            write(SHARED/'state.json',shared)
        self.sync('running','completed '+row['job'])
        return completed

    def execute(self,job,*,phase,device='gpu',method=None,target=None,seed=0,profile=None,
                role='final',exact_teacher=False,specification=None,blocked_reason=None):
        source = str(self.ensure_source())
        prior = [r for r in self.state['attempts'] if r['job']==job]
        if prior and prior[-1]['status']=='complete' and prior[-1]['source']==source:return prior[-1]
        if sum(r['source']==source for r in prior)>=2:raise CampaignStop('retry cap reached: '+job)
        available = self.remaining()
        requested_limit = (profile or {}).get('worker_wall_limit',120.)
        cpu_wall_bound=(requested_limit if (profile or {}).get('worker_cpu_limit') is not None
                        else available['cpu_core_seconds']/2.)
        wall_limit = min(requested_limit,cpu_wall_bound,available['gpu_process_seconds'] if device=='gpu' else requested_limit)
        if device!='none' and wall_limit<30.:raise CampaignStop('allocation exhausted before '+job)
        output = CAMPAIGN/f'{job}-r{len(prior)+1}'
        output.mkdir(exist_ok=False)
        row = dict(job=job,phase=phase,device=device,method=method,target=target,seed=seed,profile=profile,
            role=role,source=source,output=str(output),attempt=len(prior)+1,started_unix=time.time(),
            wall_limit=max(0.,wall_limit),cpu_limit=max(0,int(min(available['cpu_core_seconds'],wall_limit*2))))
        if (profile or {}).get('worker_cpu_limit') is not None:
            cpu_limit=profile['worker_cpu_limit']
            if type(cpu_limit) is not int or cpu_limit<1 or cpu_limit>available['cpu_core_seconds']:
                raise CampaignStop('invalid measured worker CPU reservation')
            row['cpu_limit']=cpu_limit
        if device=='none':
            write(output/'result.json',dict(status='not_executed_prerequisite',reason=blocked_reason,
                method=method,target=target,seed=seed,scientific_promotion=False))
            manifest = dict(source=source+'/source.json',status='complete',wall_seconds=0.,cpu_core_seconds=0.,
                role='decision_only_no_worker',command=[],plan=PLAN,reason=blocked_reason)
            write(output/'manifest.json',manifest)
            return self.finish({**row,'command':[]},0,manifest)
        cpu_checks = device=='cpu' and phase!='forward_reverse_teacher'
        if cpu_checks:
            command = [sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*TEST_PATHS,'--disable-warnings','--maxfail=3']
        else:
            gpu_index=-1 if phase=='forward_reverse_teacher' else (profile or {}).get('gpu_index',1)
            if gpu_index not in (-1,0,1,2):raise CampaignStop('unsupported device assignment')
            spec = {**row,'action':'preflight' if phase=='preflight' else 'trial', 'gpu':gpu_index,
                'specification':specification,'profile':profile or {},'exact_teacher':exact_teacher}
            write(output/'spec.json',spec)
            command = [sys.executable,source+'/scripts/run_neutra_scientific_campaign_master.py','worker','--spec',str(output/'spec.json')]
        env = {**os.environ,'CUDA_VISIBLE_DEVICES':'-1' if device=='cpu' else str(spec['gpu']),
            'TF_FORCE_GPU_ALLOW_GROWTH':'true','XLA_PYTHON_CLIENT_PREALLOCATE':'false','TF_CPP_MIN_LOG_LEVEL':'2',
            'TF_NUM_INTRAOP_THREADS':'2','TF_NUM_INTEROP_THREADS':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
            'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','PYTHONPATH':source,
            'BAYESFILTER_PRELOAD_CUSTOM_OP':'0','WARM_START_AUTHOR_FIXTURE':source+'/'+AUTHOR_FIXTURE}
        usage0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.monotonic()
        with (output/'stdout.log').open('w') as log:
            child = subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CPU,(row['cpu_limit'],row['cpu_limit'])))
            row.update(pid=child.pid,command=command)
            self.state['active'] = row
            self.sync('running',job)
            try:code = child.wait(timeout=wall_limit)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
                code = 124
        usage1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        path = output/'manifest.json'
        manifest = read(path) if path.exists() else {}
        manifest.update(status='complete' if code==0 else 'failed',wall_seconds=time.monotonic()-started,
            cpu_core_seconds=usage1.ru_utime+usage1.ru_stime-usage0.ru_utime-usage0.ru_stime,
            command=command,execution_cwd=source,source=source+'/source.json',
            plan=plan_for_phase(phase),
            environment=sys.executable,git_commit=read(Path(source)/'source.json')['git_commit'],
            process_cpu_accounting='parent_wait_includes_shutdown')
        if device=='cpu':
            manifest['gpu_devices_intentionally_hidden'] = True
        if cpu_checks:
            write(output/'result.json',{'status':'passed' if code==0 else 'failed','exit_code':code,'scientific_promotion':False})
        artifact_paths=output.rglob('*') if phase=='development_hmc' else output.iterdir()
        manifest['artifact_sha256'] = {str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact_paths
            if p.is_file() and p.name not in ('stdout.log','manifest.json')}
        write(path,manifest)
        complete = self.finish(row,code,manifest)
        if code!=0:raise CampaignStop('shared harness/worker failure; inspect '+str(output/'stdout.log'))
        return complete

    def result(self,job):
        rows = [r for r in self.state['attempts'] if r['job']==job and r['status']=='complete' and r['source']==str(self.source)]
        return read(Path(rows[-1]['output'])/'result.json') if rows else {}

    def select_student(self):
        for p in student_candidates():
            if all(self.result(f'student-{t}-{p["profile_id"]}').get('status')=='passed' for t in DEV_TARGETS):return p
        return None

    def select_profiles(self,student):
        selected = {}
        for method in METHODS:
            for p in profile_candidates(method):
                outcomes = [self.result(f'calibrate-{method}-{t}-{p["profile_id"]}') for t in DEV_TARGETS]
                if all(r.get('status')=='passed' and r.get('teacher_admitted') for r in outcomes):
                    selected[method] = combine_profile(p,student)
                    break
        return selected

    def price(self):
        self.write_pricing()
        print(json.dumps(read(CAMPAIGN/'pricing.json'),indent=2),flush=True)
        return 0

    def run(self):
        if self.state['status'].startswith('scientific_terminal'):
            return self.report()
        try:
            self.execute('check',phase='preflight',device='cpu')
            self.execute('preflight',phase='preflight',target='preflight',profile={})
            for t in DEV_TARGETS:
                for p in student_candidates():
                    self.execute(f'student-{t}-{p["profile_id"]}',phase='student_calibration',method='exact_teacher',
                        target=t,seed=9101 if t.endswith('two') else 9201,profile=p,role='student_calibration',
                        exact_teacher=True,specification=self.catalog[t])
            student = self.select_student()
            write(CAMPAIGN/'selected-student.json',{'profile':student,'rule':'first_cost_order_candidate_passing_both_targets',
                'scientific_promotion':False,'status':'selected' if student else 'under_calibrated'})
            for method in METHODS:
                for t in DEV_TARGETS:
                    for p in profile_candidates(method):
                        self.execute(f'calibrate-{method}-{t}-{p["profile_id"]}',phase='calibration',method=method,target=t,
                            seed=9101 if t.endswith('two') else 9201,profile=combine_profile(p,student or {}),role='calibration',
                            specification=self.catalog[t],device='gpu' if student else 'none',
                            blocked_reason='no student protocol passed both development targets')
            selected = self.select_profiles(student) if student else {}
            write(CAMPAIGN/'selected-profiles.json',{'profiles':selected,'student':student,
                'rule':'first_cost_order_candidate_passing_both_targets','scientific_promotion':False})
            for t in FINAL_TARGETS:
                self.execute(f'exact-teacher-{t}',phase='exact_teacher',method='exact_teacher',target=t,
                    seed=9001,profile=student,role='exact_teacher',exact_teacher=True,specification=self.catalog[t],
                    device='gpu' if student else 'none',blocked_reason='no calibrated student protocol')
            for method in METHODS:
                for t in FINAL_TARGETS:
                    for seed in FIT_SEEDS:
                        p = selected.get(method)
                        self.execute(f'final-{method}-{t}-s{seed}',phase='final',method=method,target=t,seed=seed,
                            profile=p,role='final',specification=self.catalog[t],device='gpu' if p else 'none',
                            blocked_reason='no complete native teacher plus student profile passed both development targets')
        except CampaignStop as error:
            self.state['stop_reason'] = str(error)
            return self.report()
        return self.report()

    def report(self):
        # No snapshot refresh and no worker launch in the terminal audit.
        write(CAMPAIGN/'state.json',self.state)
        audit = audit_campaign(CAMPAIGN,read(SHARED/'state.json'))
        if any(v < 0 for v in self.remaining().values()):
            audit['errors'].append('cumulative allocation exceeded')
            audit['status'] = 'failed'
        existing = list(CAMPAIGN.glob('audit-r*'))
        directory = CAMPAIGN/f'audit-r{len(existing)+1}'
        directory.mkdir(exist_ok=False)
        write(directory/'result.json',audit)
        self.state['terminal_audit'] = str(directory/'result.json')
        blocked = sum(c['status']=='not_executed_prerequisite' for c in audit['cells'])
        status = ('scientific_terminal_audit_failed' if audit['status']=='failed' else
                  'scientific_incomplete' if not audit['complete'] else
                  'scientific_terminal_under_calibrated' if blocked else 'scientific_terminal_screen_complete')
        self.sync(status,'review recorded candidate outcomes; '+self.state.get('stop_reason','bounded study finished'))
        summary = {'schema':'bayesfilter.neutra.scientific_campaign_result.v2','status':status,
            'scientific_promotion':False,'audit':self.state['terminal_audit'],'cells':audit['cells'],
            'not_executed_prerequisites':blocked,'resource_use_cumulative':self.used(),'remaining':self.remaining(),
            'stop_reason':self.state.get('stop_reason'), 'nonclaims':['posterior correctness','method superiority',
                'full upstream controller equivalence','HMC readiness','q20 transfer']}
        write(CAMPAIGN/'scientific-result.json',summary)
        print(json.dumps({k:summary[k] for k in ('status','not_executed_prerequisites','remaining','stop_reason')},indent=2),flush=True)
        return 0 if audit['complete'] and audit['status']=='passed' else 1


REMEDY_SEEDS = (11, 37, 73)
REMEDY_PROFILE = dict(profile_id='matched_fixed_fresh_w64', width=64,batch=64,
    learning_rate=.001,updates=8192,continuation_updates=2048,jit_compile=True,
    worker_wall_limit=360.)


def remedy_jobs():
    for target in DEV_TARGETS:
        for seed in REMEDY_SEEDS:
            yield f'matched-{target}-s{seed}',target,seed


def optimizer_jobs():
    for target in DEV_TARGETS:
        for seed in REMEDY_SEEDS:
            yield f'optimizer-{target}-s{seed}',target,seed


class RemedyController(Controller):
    """Same execution/budget machinery, explicit repaired fitting phase order."""

    def sync(self,status,next_action):
        self.state.update(status=status,next_action=next_action,remaining=self.remaining(),updated_unix=time.time())
        write(CAMPAIGN/'state.json',self.state)
        write(CAMPAIGN/'next-phase.json',dict(phase=status,next_action=next_action,remaining=self.remaining(),
            resume_command='bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_scientific_campaign.sh resume'))
        latest = {r['job']:r for r in self.state['attempts']}
        write(CAMPAIGN/'matrix.json',{'rows':[dict(job=job,target=t,seed=s,
            status=latest.get(job,{}).get('status','not_run'),output=latest.get(job,{}).get('output'))
            for job,t,s in (*remedy_jobs(),*optimizer_jobs(),
                *((f'naf-{t}-s{s}',t,s) for t in DEV_TARGETS for s in REMEDY_SEEDS),
                *((f'development-hmc-{t}-s{s}',t,s) for t in DEV_TARGETS for s in REMEDY_SEEDS))],
                'scientific_promotion':False})
        self.write_pricing()

    def write_pricing(self):
        audit_path = CAMPAIGN/'history-audit-r1.json'
        audit = read(audit_path) if audit_path.exists() else {}
        prices = audit.get('r3_prices',[])
        large = [r for r in prices if r['profile'].get('width')==64]
        # Two complete baseline workers, plus three quarter-length continuation
        # endpoints priced as three further complete workers: conservative until
        # the first full paired job measures shared setup and compilation.
        baseline = max((r['worker_wall_seconds'] for r in large),default=0.)
        complete = {r['job']:r for r in self.state['attempts'] if r['status']=='complete'}
        pending = sum(job not in complete for job,_,_ in remedy_jobs())
        cap = REMEDY_PROFILE['worker_wall_limit']
        write(CAMPAIGN/'pricing.json',{
            'remaining':self.remaining(),'used_cumulative':self.used(),
            'measured_r3_full_worker_max_seconds':baseline,
            'forecast_per_paired_job_seconds':5*baseline,
            'forecast_basis':'five complete r3 endpoints; new joint compilation remains uncertain',
            'worker_hard_limit_seconds':cap,'pending_paired_jobs':pending,
            'complete_primary_reservation':{'gpu_process_seconds':pending*cap,'cpu_core_seconds':pending*cap*2},
            'initial_jobs':6,'endpoints_per_job':5,'fit_seeds':list(REMEDY_SEEDS),
            'source_reproduction_and_posterior_confirmation':'not_in_initial_fit_forecast',
            'full_proposal_affordability':'not_established', 'scientific_promotion':False})

    def run(self):
        completed={r['job'] for r in self.state['attempts'] if r['status']=='complete'}
        if self.state['status'].startswith('remedy_fit_complete') or all(job in completed for job,_,_ in remedy_jobs()):
            return self.report()
        try:
            if not (CAMPAIGN/'history-audit-r1.json').exists():
                raise CampaignStop('P0 historical audit missing')
            self.execute('check',phase='preflight',device='cpu')
            self.execute('preflight',phase='preflight',target='preflight',profile={})
            self.write_pricing()
            reservation = read(CAMPAIGN/'pricing.json')['complete_primary_reservation']
            if any(reservation[k]>self.remaining()[k] for k in LIMITS):
                raise CampaignStop('remaining allocation cannot fund complete matched comparison')
            for job,target,seed in remedy_jobs():
                self.execute(job,phase='matched_fit',method='exact_teacher',target=target,seed=seed,
                    profile=REMEDY_PROFILE,role='development_diagnostic',specification=self.catalog[target])
        except CampaignStop as error:
            self.state['stop_reason']=str(error)
        return self.report()

    def report(self):
        rows,errors = [],[]
        for job,target,seed in remedy_jobs():
            matches = [r for r in self.state['attempts'] if r['job']==job and r['status']=='complete']
            if not matches:
                rows.append(dict(job=job,target=target,seed=seed,status='not_complete'))
                continue
            attempt=matches[-1]
            output=Path(attempt['output'])
            result=read(output/'result.json')
            manifest=read(output/'manifest.json')
            for name,expected in manifest.get('artifact_sha256',{}).items():
                if hashlib.sha256((output/name).read_bytes()).hexdigest()!=expected:
                    errors.append(f'{job}: changed {name}')
            source=read(Path(attempt['source'])/'source.json')
            for name,expected in source['sha256'].items():
                if hashlib.sha256((Path(attempt['source'])/name).read_bytes()).hexdigest()!=expected:
                    errors.append(f'{job}: changed source {name}')
            rows.append(dict(job=job,target=target,seed=seed,status=result['status'],
                result=str(output/'result.json'),endpoints={name:{
                    'coarse_screen_passed':value['passed'],
                    'forward_kl':value['heldout']['forward_kl_estimate'],
                    'reverse_kl':value['heldout']['reverse_kl_estimate'],
                    'mass_error':value['heldout']['maximum_responsibility_discrepancy'],
                    'summary_z_max':value['heldout']['summary_z_max'],
                    'transport_hash':value['transport_hash']}
                    for name,value in result['endpoints'].items()}))
        complete=all(r['status']=='comparison_complete' for r in rows) and not errors
        next_action=('review paired fits; choose priced optimization/representation repair or native transfer'
                     if complete else self.state.get('stop_reason','complete missing paired fits'))
        self.sync('remedy_fit_complete' if complete else 'remedy_fit_incomplete',next_action)
        result={'status':self.state['status'],'rows':rows,'errors':errors,'remaining':self.remaining(),
            'next_action':next_action,'scientific_promotion':False,'posterior_confirmation':'not_run',
            'source_reproduction':'not_run','method_ranking':'not_established'}
        write(CAMPAIGN/'remedy-result.json',result)
        print(json.dumps({k:result[k] for k in ('status','errors','remaining','next_action')},indent=2),flush=True)
        return 0 if complete else 1

    def run_optimizer_repair(self):
        completed={r['job']:r for r in self.state['attempts'] if r['status']=='complete'}
        pending=[(job,t,s) for job,t,s in optimizer_jobs() if job not in completed]
        if not pending:
            return self.optimizer_report()
        try:
            if not all(job in completed for job,_,_ in remedy_jobs()):
                raise CampaignStop('complete fixed/fresh comparison required before optimizer repair')
            reservation={'gpu_process_seconds':180.*len(pending),'cpu_core_seconds':360.*len(pending)}
            write(CAMPAIGN/'optimizer-pricing.json',{'pending_jobs':len(pending),'reservation':reservation,
                'remaining':self.remaining(),'basis':'180-second bound from measured 142-second five-endpoint jobs'})
            if any(reservation[k]>self.remaining()[k] for k in LIMITS):
                raise CampaignStop('complete optimizer comparison exceeds remaining allocation')
            self.execute('check',phase='preflight',device='cpu')
            for job,target,seed in pending:
                parent=completed[f'matched-{target}-s{seed}']['output']
                profile={**REMEDY_PROFILE,'profile_id':'matched_forward_lr_repair','parent':parent,
                         'continuation_updates':0,'worker_wall_limit':180.}
                self.execute(job,phase='optimizer_repair',method='exact_teacher',target=target,seed=seed,
                    profile=profile,role='development_diagnostic',specification=self.catalog[target])
        except CampaignStop as error:
            self.state['stop_reason']=str(error)
        return self.optimizer_report()

    def optimizer_report(self):
        rows,errors=[],[]
        for job,target,seed in optimizer_jobs():
            matches=[r for r in self.state['attempts'] if r['job']==job and r['status']=='complete']
            if not matches:
                rows.append(dict(job=job,target=target,seed=seed,status='not_complete'))
                continue
            attempt=matches[-1]
            output=Path(attempt['output'])
            manifest=read(output/'manifest.json')
            for name,expected in manifest['artifact_sha256'].items():
                if hashlib.sha256((output/name).read_bytes()).hexdigest()!=expected:
                    errors.append(f'{job}: changed {name}')
            source=read(Path(attempt['source'])/'source.json')
            for name,expected in source['sha256'].items():
                if hashlib.sha256((Path(attempt['source'])/name).read_bytes()).hexdigest()!=expected:
                    errors.append(f'{job}: changed source {name}')
            result=read(output/'result.json')
            rows.append(dict(job=job,target=target,seed=seed,status=result['status'],result=str(output/'result.json'),
                endpoints={name:{'coarse_screen_passed':v['passed'],
                    'forward_kl':v['heldout']['forward_kl_estimate'],
                    'reverse_kl':v['heldout']['reverse_kl_estimate'],
                    'mass_error':v['heldout']['maximum_responsibility_discrepancy'],
                    'summary_z_max':v['heldout']['summary_z_max']}
                    for name,v in result['endpoints'].items()}))
        complete=all(r['status']=='comparison_complete' for r in rows) and not errors
        next_action=('price bounded representation comparison if density remains inadequate; otherwise native transfer'
                     if complete else self.state.get('stop_reason','complete missing optimizer pairs'))
        self.sync('remedy_optimizer_complete' if complete else 'remedy_optimizer_incomplete',next_action)
        result=dict(status=self.state['status'],rows=rows,errors=errors,remaining=self.remaining(),
            next_action=next_action,scientific_promotion=False,posterior_confirmation='not_run',method_ranking='not_established')
        write(CAMPAIGN/'optimizer-result.json',result)
        print(json.dumps({k:result[k] for k in ('status','errors','remaining','next_action')},indent=2),flush=True)
        return 0 if complete else 1

    def run_representation_check(self):
        completed={r['job']:r for r in self.state['attempts'] if r['status']=='complete'}
        if not all(job in completed for job,_,_ in optimizer_jobs()):
            raise CampaignStop('optimizer comparison incomplete')
        if all(f'naf-{t}-s{s}' in completed for t in DEV_TARGETS for s in REMEDY_SEEDS):
            return self.representation_report()
        optimizer=read(CAMPAIGN/'optimizer-result.json')
        if all(row['endpoints']['lr-lower']['coarse_screen_passed'] for row in optimizer['rows']):
            self.sync('remedy_native_transfer_pending','price native teacher transfer and fixed-map posterior verification')
            return 0
        try:
            self.execute('check',phase='preflight',device='cpu',profile={'worker_wall_limit':240.})
            failures=[r for r in self.state['attempts'] if r['job'].startswith('naf-')
                and r['status']=='failed' and r['job'] not in completed]
            if failures:
                failed=failures[-1]
                diagnostic='repaired-replay-'+failed['job']+'-r'+str(failed['attempt'])
                replay=self.execute(diagnostic,phase='representation_debug',method='exact_teacher',
                    target=failed['target'],seed=failed['seed'],
                    profile={**failed['profile'],'failed_output':failed['output'],'worker_wall_limit':240.},
                    role='numerical_repair_verification_only',specification=self.catalog[failed['target']])
                evidence=read(Path(replay['output'])/'result.json')
                if not (evidence['replay']['finite'] and evidence['replay']['updates']==4096
                        and evidence['objective']['valid']
                        and all(not row['invalid_rows'] for row in evidence['inverse_summary'].values())):
                    raise CampaignStop('saved NAF failure remains after repair; inspect replay')
                self.state.pop('stop_reason',None)
            profile={**REMEDY_PROFILE,'profile_id':'naf_dsf_pricing','kind':'naf_dsf',
                     'updates':1024,'continuation_updates':0,'pricing_only':True,'worker_wall_limit':240.}
            self.execute('representation-price',phase='representation',method='exact_teacher',
                target=DEV_TARGETS[1],seed=11,profile=profile,role='pricing_only',specification=self.catalog[DEV_TARGETS[1]])
            attempt=[r for r in self.state['attempts'] if r['job']=='representation-price' and r['status']=='complete'][-1]
            pricing=read(Path(attempt['output'])/'result.json')
            warm=pricing['training'][-1]
            per_update=warm['wall_seconds']/warm['updates']
            # Twice total pricing overhead includes both architecture-stage
            # compilations and endpoint checks; warmed update work is additional.
            limit=math.ceil(2*attempt['wall_seconds']+16384*per_update)
            pending=[(f'naf-{t}-s{s}',t,s) for t in DEV_TARGETS for s in REMEDY_SEEDS
                     if f'naf-{t}-s{s}' not in completed]
            reserve={'gpu_process_seconds':len(pending)*limit,'cpu_core_seconds':len(pending)*limit*2}
            write(CAMPAIGN/'representation-pricing.json',dict(measured_pricing_worker_seconds=attempt['wall_seconds'],
                warmed_seconds_per_update=per_update,worker_limit=limit,reservation=reserve,
                remaining=self.remaining(),pending_jobs=len(pending),complete_recipe_updates=16384,
                forecast_uncertainty='different data shapes and optimizer graph compilation'))
            if any(reserve[k]>self.remaining()[k] for k in LIMITS):
                raise CampaignStop('complete NAF comparison requires measured allocation extension; see representation-pricing.json')
            for job,target,seed in pending:
                profile={**REMEDY_PROFILE,'profile_id':'naf_dsf_matched','kind':'naf_dsf','pricing_only':False,
                    'worker_wall_limit':limit,'parent':completed[f'matched-{target}-s{seed}']['output'],
                    'optimizer_parent':completed[f'optimizer-{target}-s{seed}']['output']}
                self.execute(job,phase='representation',method='exact_teacher',target=target,seed=seed,
                    profile=profile,role='development_diagnostic',specification=self.catalog[target])
        except CampaignStop as error:
            self.state['stop_reason']=str(error)
        return self.representation_report()

    def representation_report(self):
        rows,errors=[],[]
        for target in DEV_TARGETS:
            for seed in REMEDY_SEEDS:
                job=f'naf-{target}-s{seed}'
                matches=[r for r in self.state['attempts'] if r['job']==job and r['status']=='complete']
                if not matches:
                    rows.append(dict(job=job,target=target,seed=seed,status='not_complete'))
                    continue
                attempt=matches[-1];output=Path(attempt['output'])
                manifest=read(output/'manifest.json')
                for name,expected in manifest['artifact_sha256'].items():
                    if hashlib.sha256((output/name).read_bytes()).hexdigest()!=expected:
                        errors.append(f'{job}: changed {name}')
                source=read(Path(attempt['source'])/'source.json')
                for name,expected in source['sha256'].items():
                    if hashlib.sha256((Path(attempt['source'])/name).read_bytes()).hexdigest()!=expected:
                        errors.append(f'{job}: changed source {name}')
                result=read(output/'result.json')
                rows.append(dict(job=job,target=target,seed=seed,status=result['status'],result=str(output/'result.json'),
                    endpoints={name:{'coarse_screen_passed':v['passed'],
                        'forward_kl':v['heldout']['forward_kl_estimate'],
                        'reverse_kl':v['heldout']['reverse_kl_estimate'],
                        'mass_error':v['heldout']['maximum_responsibility_discrepancy'],
                        'summary_z_max':v['heldout']['summary_z_max']} for name,v in result['endpoints'].items()}))
        complete=all(r['status']=='comparison_complete' for r in rows) and not errors
        next_action=('review representation; price native transfer or next fit repair'
                     if complete else self.state.get('stop_reason','complete missing representation pairs'))
        self.sync('remedy_representation_complete' if complete else 'remedy_representation_incomplete',next_action)
        write(CAMPAIGN/'representation-result.json',dict(status=self.state['status'],rows=rows,errors=errors,
            next_action=next_action,remaining=self.remaining(),scientific_promotion=False))
        print(json.dumps(dict(status=self.state['status'],remaining=self.remaining(),next_action=next_action),indent=2),flush=True)
        return 0 if complete else 1


    def run_development_hmc(self):
        representation=read(CAMPAIGN/'representation-result.json')
        if representation['status']!='remedy_representation_complete' or representation['errors']:
            raise CampaignStop('complete numerically valid representation comparison required')
        latest={r['job']:r for r in self.state['attempts'] if r['status']=='complete'}
        eligible=[r for r in representation['rows'] if r['endpoints']['lr-lower']['coarse_screen_passed']]
        if not eligible:
            self.sync('remedy_geometry_repair_pending','no endpoint passed fit screen; price next representation/geometry repair')
            return 0
        if all(f"development-hmc-{r['target']}-s{r['seed']}" in latest for r in eligible):
            return self.development_report(representation)
        try:
            self.execute('check',phase='preflight',device='cpu',profile={'worker_wall_limit':240.})
            control_job='development-hmc-gaussian-control'
            control=latest.get(control_job)
            if control is None:
                control=self.execute(control_job,phase='development_hmc',method='identity_control',
                    target='gaussian_control',seed=11,role='numerical_control',
                    profile={'control':True,'worker_wall_limit':600.,'profile_id':'gaussian_control'},
                    specification={'kind':'gaussian','mean':[0.,0.],'covariance':[[1.,0.],[0.,1.]]})
            if not read(Path(control['output'])/'result.json')['passed']:
                raise CampaignStop('shared Gaussian HMC control not passed; investigate before fitted maps')
            pending=[r for r in eligible if f"development-hmc-{r['target']}-s{r['seed']}" not in latest]
            def execute_map(row,limit):
                target,seed=row['target'],row['seed']
                return self.execute(f'development-hmc-{target}-s{seed}',phase='development_hmc',
                    method='exact_teacher_naf',target=target,seed=seed,role='development_diagnostic',
                    profile={'parent':str(Path(row['result']).parent),'target_label':target,
                        'target_index':DEV_TARGETS.index(target),'worker_wall_limit':limit,
                        'profile_id':'development_frozen_naf_hmc'},specification=self.catalog[target])
            price_path=CAMPAIGN/'development-hmc-pricing.json'
            if pending and not price_path.exists():
                if self.remaining()['gpu_process_seconds']<1800 or self.remaining()['cpu_core_seconds']<3600:
                    raise CampaignStop('insufficient budget for complete first development HMC map')
                first=execute_map(pending.pop(0),1800.)
                result=read(Path(first['output'])/'result.json')
                # A complete first map measures real compilation, tuning and
                # all attempted sequential checks. Double observed cost for
                # target variation, with no lower ceiling than the first map.
                limit=max(1800,math.ceil(2*first['wall_seconds']))
                write(price_path,dict(measured_worker_seconds=first['wall_seconds'],
                    first_result=str(Path(first['output'])/'result.json'),
                    tuning_seconds=result.get('tuning_wall_seconds'),worker_limit=limit,
                    ceiling_basis='max(first containment bound, twice observed full worker cost)',
                    target_variation_uncertainty='three-mode cost unmeasured; candidate caps remain unchanged'))
            if pending:
                limit=read(price_path)['worker_limit']
                reservation={'gpu_process_seconds':len(pending)*limit,'cpu_core_seconds':2*len(pending)*limit}
                pricing=read(price_path)
                write(price_path,{**pricing,'remaining_before_queue':self.remaining(),
                    'pending_jobs':len(pending),'reservation':reservation})
                if any(reservation[k]>self.remaining()[k] for k in LIMITS):
                    raise CampaignStop('complete remaining development scope exceeds allocation')
                for row in pending:
                    execute_map(row,limit)
        except CampaignStop as error:
            self.state['stop_reason']=str(error)
            self.sync('development_hmc_incomplete',str(error))
            return 1
        return self.development_report(representation)

    def development_report(self,representation):
        """A completed resume audits saved evidence; it launches no workers."""
        rows,errors,checked_sources=[],[],set()
        latest={r['job']:r for r in self.state['attempts'] if r['status']=='complete'}
        for row in representation['rows']:
            job=f"development-hmc-{row['target']}-s{row['seed']}"
            attempt=latest.get(job)
            result=read(Path(attempt['output'])/'result.json') if attempt else {}
            if attempt:
                output=Path(attempt['output'])
                for name,expected in read(output/'manifest.json').get('artifact_sha256',{}).items():
                    if hashlib.sha256((output/name).read_bytes()).hexdigest()!=expected:
                        errors.append(f'{job}: changed {name}')
                source=attempt.get('source')
                if source and source not in checked_sources:
                    for name,expected in read(Path(source)/'source.json')['sha256'].items():
                        if hashlib.sha256((Path(source)/name).read_bytes()).hexdigest()!=expected:
                            errors.append(f'{job}: changed source {name}')
                    checked_sources.add(source)
            rows.append({'target':row['target'],'seed':row['seed'],'fit_passed':row['endpoints']['lr-lower']['coarse_screen_passed'],
                'status':result.get('status','fit_screen_failed'),'passed':result.get('passed',False),
                'result':str(Path(attempt['output'])/'result.json') if attempt else None})
        if errors:
            action='repair invalid development evidence: '+'; '.join(errors)
            self.state['stop_reason']=action
        else:
            self.state.pop('stop_reason',None)
            action='review downstream validation; price native-teacher transfer for viable maps or geometry repair for failed maps'
        self.sync('development_hmc_invalid' if errors else 'development_hmc_complete',action)
        write(CAMPAIGN/'development-hmc-result.json',dict(rows=rows,status=self.state['status'],
            errors=errors,
            remaining=self.remaining(),next_action=action,method_ranking='not_established',scientific_promotion=False,
            scope='development exact-teacher fixed-map validation; no final generalization or q20 claim'))
        return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('check','preflight','price','run','resume','status','worker','attribution',
                                        'forward-reverse-plan','forward-reverse','forward-reverse-smoke'))
    parser.add_argument('--spec')
    args = parser.parse_args()
    if args.action=='worker':return worker(args.spec)
    if args.action=='status':
        if not (CAMPAIGN/'state.json').exists():
            print('Scientific revision 3 not initialized')
            return 0
        state = read(CAMPAIGN/'state.json')
        print(json.dumps({k:state.get(k) for k in ('status','next_action','remaining','active','terminal_audit')},indent=2))
        return 0
    with (SHARED/'master.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller = RemedyController()
        controller.settle_pending_charges()
        if args.action=='forward-reverse-plan':
            from neutra_forward_reverse_campaign import program
            # Device-free refresh settles verification charges but launches no worker.
            controller.sync(controller.state['status'],controller.state['next_action'])
            result=program(controller.remaining())
            write(CAMPAIGN/'forward-reverse-program.json',result)
            print(json.dumps(result,indent=2)); return 0
        smoke_path=CAMPAIGN/'forward-reverse-smoke.json'
        if args.action=='forward-reverse-smoke' or (args.action in ('run','resume') and
            smoke_path.exists() and read(smoke_path)['status']!='passed'):
            from neutra_forward_reverse_campaign import smoke
            return smoke(controller)
        if args.action=='forward-reverse' or (args.action in ('run','resume') and
            ((CAMPAIGN/'forward-reverse-state.json').exists() or
             controller.state['status'] in ('attribution_complete','forward_reverse_prepared'))):
            from neutra_forward_reverse_campaign import run_campaign
            return run_campaign(controller)
        if args.action=='attribution':
            from neutra_attribution_campaign import run_campaign
            return run_campaign(controller)
        if args.action=='price':return controller.price()
        if args.action in ('check','preflight'):
            controller.execute(args.action,phase='preflight',device='cpu' if args.action=='check' else 'gpu',
                target=None if args.action=='check' else 'preflight',profile={})
            return 0
        code=controller.run()
        if code==0:code=controller.run_optimizer_repair()
        if code==0:code=controller.run_representation_check()
        return controller.run_development_hmc() if code==0 else code


if __name__=='__main__':
    raise SystemExit(main())
