#!/usr/bin/env python3
"""Resumable bounded four-method campaign; framework imports stay in workers."""
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

ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = Path('/home/ubuntu/python/BayesFilter')
PLAN = 'docs/plans/bayesfilter-neutra-rare-region-master-2026-10-02.md'
CAMPAIGN = LIVE_ROOT/'docs/plans/artifacts/neutra-rare-region-2026-10-02/campaign-r1'
SHARED = LIVE_ROOT/'docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1'
METHODS = ('umbrella', 'importance', 'splitting', 'local_maps')
RESOURCE_KEYS = ('gpu_process_seconds', 'cpu_core_seconds')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def config():
    return {'targets': ['mixture', 'warped_mixture'], 'methods': list(METHODS),
        'replication_seeds': [101, 211, 307, 401, 503, 601, 701, 809],
        'training_seeds': [11, 37], 'particles': 4096, 'widths': [16, 32],
        'learning_rates': [.0003, .001], 'batch_size': 256, 'pilot_updates': 256,
        'forward_updates': 1024, 'reverse_rungs': [256, 1024], 'gpu': 1,
        'gpu_process_seconds': 7200., 'cpu_core_seconds': 14400.,
        'cpu_workers': 2, 'cpu_threads': 2, 'worker_wall_seconds': 900.,
        'hmc_wall_seconds': 240., 'attempts_per_job': 2, 'plan': PLAN,
        'result_file': 'docs/plans/bayesfilter-neutra-rare-region-results-2026-10-02.md',
        'hmc': {'leapfrogs': [3, 9, 18], 'initial_epsilon': .5,
            'max_candidates': 18, 'work_units': 72, 'job_wall_seconds': 220.,
            'diagnostic_profile': 'moments_regions_shape_v4',
            'temporal_conflict_method': 'paired_chain_block_contrasts_v1',
            'posterior_member_limit': 2, 'measurement_num_results': 64,
            'verification_num_results': 64, 'evidence_rungs': [1, 2], 'refinement_rounds': 1}}


def snapshot(root):
    paths = list((LIVE_ROOT/'bayesfilter').rglob('*.py'))
    paths += [LIVE_ROOT/'scripts/run_neutra_rare_region_master.py', LIVE_ROOT/PLAN,
              LIVE_ROOT/'tests/test_neutra_rare_regions.py', LIVE_ROOT/'docs/chapters/ch26e_rare_regions_transport.tex']
    for name in ('scripts/run_neutra_controlled_repair_master.py', 'tests/test_neutra_controlled_repair.py'):
        if (LIVE_ROOT/name).exists(): paths.append(LIVE_ROOT/name)
    hashes = {str(p.relative_to(LIVE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    dest = root/('source-'+signature[:12])
    if not dest.exists():
        dest.mkdir()
        for name in hashes:
            output = dest/name; output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(LIVE_ROOT/name, output)
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=LIVE_ROOT, text=True).strip()
        write(dest/'source.json', {'git_commit': revision, 'sha256': hashes,
            'signature': signature, 'dirty_source_preserved': True})
    for name, digest in hashes.items():
        if hashlib.sha256((dest/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('frozen source was changed')
    return dest


def worker(spec_path):
    spec = read(spec_path); cfg = spec['config']; out = Path(spec['output'])
    gpu = spec['device'] == 'gpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu']) if gpu else '-1'
    if gpu and os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true':
        raise RuntimeError('memory growth must be enabled before framework import')
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(spec['cpu_threads'])
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = str(spec['cpu_threads'])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    resource.setrlimit(resource.RLIMIT_CPU, (spec['cpu_limit'], spec['cpu_limit']))
    start = time.monotonic()
    initial = resource.getrusage(resource.RUSAGE_SELF)
    manifest = {'command': [sys.executable, str(Path(__file__).resolve()), 'worker', '--spec', str(spec_path)],
        'plan': PLAN, 'result_file': cfg['result_file'], 'environment': sys.executable,
        'data_version': 'WarmStartTarget.signature in phase artifact', 'seed': spec.get('seed'),
        'replication_seeds': cfg['replication_seeds'], 'source_manifest': str(ROOT/'source.json'),
        'git_commit': read(ROOT/'source.json')['git_commit'] if (ROOT/'source.json').exists() else 'unfrozen_test',
        'output': str(out), 'device': spec['device'], 'gpu_devices_intentionally_hidden': not gpu,
        'jit_compile': True, 'dtype': 'float64 analytic reference campaign', 'tf32': True,
        'input_sha256': {}, 'started_unix': time.time()}
    for field in ('prepared', 'local', 'teacher', 'training'):
        if spec.get(field):
            directory = Path(spec[field])
            for p in directory.rglob('*'):
                if p.is_file() and p.suffix in ('.json', '.tensor'):
                    manifest['input_sha256'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    write(out/'manifest.json', manifest)
    code = 0
    try:
        sys.path.insert(0, str(ROOT))
        import tensorflow as tf
        if gpu:
            from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
            manifest['memory_policy'] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
            manifest['trust_basis'] = 'trusted_escalated_campaign_wrapper'
        else:
            manifest['memory_policy'] = {'mode': 'cpu_only_gpu_hidden'}
        tf.config.experimental.enable_tensor_float_32_execution(True)
        manifest['tensorflow_version'] = tf.__version__
        write(out/'manifest.json', manifest)
        from bayesfilter.testing import neutra_rare_region_phases as phases
        action = spec['action']; target = spec['target']
        if action == 'prepare':
            result = phases.prepare(target, out, cfg)
        elif action == 'local_fit':
            result = phases.local_fit(target, spec['prepared'], out, cfg)
        elif action == 'method':
            result = phases.run_method(target, spec['method'], spec['prepared'], spec.get('local'), out, cfg, spec.get('repair', 0))
        elif action == 'fit':
            result = phases.global_fit(target, spec['teacher'], spec['prepared'], out, spec['seed'], cfg)
        elif action == 'qualify':
            from bayesfilter.testing.neutra_warm_start_qualification import qualify_shortlist
            cfg = {**cfg, 'hmc': {**cfg['hmc'], 'shortlist_wall_seconds': 220.}}
            result = qualify_shortlist(target, spec['seed'], spec['prepared'], spec['training'], out, cfg)
        else:
            raise ValueError('unrecognized phase')
        phases.write_json(out/'result.json', result)
        manifest['status'] = 'complete'
        if gpu:
            manifest['allocator'] = tf.config.experimental.get_memory_info('GPU:0')
    except Exception as exc:
        traceback.print_exc()
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        code = 1
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        manifest['wall_seconds'] = time.monotonic()-start
        manifest['cpu_core_seconds'] = usage.ru_utime+usage.ru_stime-initial.ru_utime-initial.ru_stime
        manifest['gpu_process_seconds'] = manifest['wall_seconds'] if gpu else 0.
        manifest['finished_unix'] = time.time()
        write(out/'manifest.json', manifest)
    return code


class Controller:
    def worker_entry(self):
        return 'scripts/run_neutra_rare_region_master.py'

    def __init__(self):
        self.root = CAMPAIGN; self.root.mkdir(parents=True, exist_ok=True)
        self.cfg = read(self.root/'config.json') if (self.root/'config.json').exists() else config()
        write(self.root/'config.json', self.cfg)
        self.state = read(self.root/'state.json') if (self.root/'state.json').exists() else {
            'schema': 'bayesfilter.neutra.rare_region_master.v1', 'jobs': {}, 'active': {},
            'created_unix': time.time(), 'status': 'prepared'}
        self.source = snapshot(self.root)
        self.shared_cfg = read(SHARED/'config.json')
        self.shared = read(SHARED/'state.json')
        if self.shared.get('active_job'):
            raise RuntimeError('shared campaign has active/unreconciled work')
        self.recover()

    def remaining(self):
        rows = [r for attempts in self.state['jobs'].values() for r in attempts]
        return {k: self.cfg[k]-sum(r.get(k, 0.) for r in rows) for k in RESOURCE_KEYS}

    def sync(self, phase, next_action=None):
        self.state.update(status=phase, next_action=next_action, remaining=self.remaining(), updated_unix=time.time())
        write(self.root/'state.json', self.state)
        write(self.root/'next-phase.json', {'plan': PLAN, 'status': phase, 'next_action': next_action,
            'remaining': self.remaining(), 'active': self.state['active'],
            'resume_command': self.cfg.get('resume_command','bash /home/ubuntu/python/BayesFilter/scripts/run_neutra_rare_region_campaign.sh resume')})

    def recover(self):
        for name, row in list(self.state['active'].items()):
            pid = row.get('pid')
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f'prior worker {pid} is still live; inspect status, do not duplicate')
            manifest = Path(row['output'])/'manifest.json'
            done = read(manifest) if manifest.exists() else {}
            if done.get('status') == 'complete' and (Path(row['output'])/'result.json').exists():
                code = 0
            else:
                code = 125
            self.finish(name, row, code, done)

    def finish(self, name, row, code, manifest):
        wall = manifest.get('wall_seconds', min(time.time()-row['started_unix'], row['wall_limit']))
        cpu = manifest.get('cpu_core_seconds', row['cpu_limit'])
        record = {**row, 'exit_code': code, 'status': 'complete' if code == 0 else 'failed',
            'wall_seconds': wall, 'gpu_process_seconds': wall if row['device'] == 'gpu' else 0.,
            'cpu_core_seconds': cpu, 'cpu_accounting_conservative': 'cpu_core_seconds' not in manifest}
        self.state['jobs'].setdefault(name, []).append(record)
        self.state['active'].pop(name, None)
        charge_id = 'rare-region-20261002-'+name+'-r'+str(record['attempt'])
        if not any(x['job'] == charge_id for x in self.shared['attempts']):
            self.shared['attempts'].append({'job': charge_id, 'phase': 'rare_region',
                'output': row['output'], 'status': record['status'], 'exit_code': code,
                **{k: record[k] for k in RESOURCE_KEYS}, 'wall_seconds': wall})
            write(SHARED/'state.json', self.shared)
        self.sync('phase_complete' if code == 0 else 'repair_required', name)
        print(json.dumps({'event': 'finish', 'job': name, 'status': record['status'],
            'wall_seconds': wall, 'remaining': self.remaining()}), flush=True)

    def done(self, name):
        rows = self.state['jobs'].get(name, [])
        return Path(rows[-1]['output']) if rows and rows[-1]['status'] == 'complete' else None

    def batch(self, specs):
        pending = [s for s in specs if self.done(s['name']) is None]
        running = {}
        while pending or running:
            while pending and len(running) < (self.cfg.get('gpu_workers',1) if pending[0]['device'] == 'gpu' else self.cfg['cpu_workers']):
                spec = pending.pop(0); name = spec['name']
                attempt = len(self.state['jobs'].get(name, []))+1
                if attempt > self.cfg['attempts_per_job']:
                    raise RuntimeError(f'attempt ceiling reached for {name}; inspect recorded error')
                rem = self.remaining()
                reservations = {k: sum(r[k] for _, r, _ in running.values()) for k in RESOURCE_KEYS}
                shared_left = {k: self.shared_cfg[k]-sum(r.get(k, 0.) for r in self.shared['attempts']) for k in RESOURCE_KEYS}
                available = {k: min(rem[k], shared_left[k])-reservations[k] for k in RESOURCE_KEYS}
                threads = self.cfg['cpu_threads']
                wall = min(self.cfg['hmc_wall_seconds'] if spec['action'] == 'qualify' else self.cfg['worker_wall_seconds'],
                           available['cpu_core_seconds']/max(threads*2, 1),
                           available['gpu_process_seconds'] if spec['device'] == 'gpu' else math.inf)
                if wall < 20:
                    if running:
                        pending.insert(0, spec)
                        break
                    self.sync('budget_exhausted', name)
                    raise RuntimeError('insufficient bounded worker budget')
                output = self.root/'attempts'/f'{name}-r{attempt}'
                output.mkdir(parents=True, exist_ok=False)
                cpu_limit = max(1, math.floor(min(available['cpu_core_seconds'], wall*threads*2)))
                row = {**spec, 'output': str(output), 'attempt': attempt, 'wall_limit': wall,
                    'cpu_limit': cpu_limit, 'cpu_threads': threads, 'started_unix': time.time(), 'source': str(self.source)}
                write(output/'spec.json', {**row, 'config': self.cfg})
                command = [sys.executable, str(self.source/self.worker_entry()),
                    'worker', '--spec', str(output/'spec.json')]
                log = (output/'process.log').open('w')
                env = os.environ.copy(); env.update(TF_FORCE_GPU_ALLOW_GROWTH='true',
                    CUDA_VISIBLE_DEVICES=str(self.cfg['gpu']) if spec['device'] == 'gpu' else '-1',
                    PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
                process = subprocess.Popen(command, cwd=self.source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row.update(pid=process.pid, command=command)
                self.state['active'][name] = row
                reserve = {'cpu_core_seconds': cpu_limit, 'gpu_process_seconds': wall if spec['device'] == 'gpu' else 0.}
                running[name] = (process, reserve, log)
                self.sync('running', name)
                print(json.dumps({'event': 'start', 'job': name, 'pid': process.pid, 'wall_limit': wall}), flush=True)
            for name, (process, reserve, log) in list(running.items()):
                row = self.state['active'][name]
                if process.poll() is None and time.time()-row['started_unix'] > row['wall_limit']:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                if process.poll() is not None:
                    log.close(); del running[name]
                    manifest_path = Path(row['output'])/'manifest.json'
                    manifest = read(manifest_path) if manifest_path.exists() else {}
                    self.finish(name, row, process.returncode, manifest)
            if running:
                time.sleep(1)
        failed = [s['name'] for s in specs if self.done(s['name']) is None]
        if failed:
            raise RuntimeError('localized infrastructure/implementation repair required: '+', '.join(failed))

    def run(self):
        def job(action, target, name, device='cpu', **extra):
            return {'action': action, 'target': target, 'name': name, 'device': device, **extra}
        targets = self.cfg['targets']
        self.batch([job('prepare', t, 'prepare-'+t) for t in targets])
        self.batch([job('local_fit', t, 'local-'+t, 'gpu', prepared=str(self.done('prepare-'+t))) for t in targets])
        methods = [job('method', t, f'{t}-{m}-v0', method=m, prepared=str(self.done('prepare-'+t)),
            local=str(self.done('local-'+t)), repair=0) for t in targets for m in METHODS]
        self.batch(methods)
        repairs = []
        for spec in methods:
            result = read(self.done(spec['name'])/'result.json')
            if not result['screen']['passed']:
                repairs.append({**spec, 'name': spec['name'][:-1]+'1', 'repair': 1})
        self.sync('scientific_repair_planned', [s['name'] for s in repairs])
        self.batch(repairs)
        teachers = {}
        for t in targets:
            for m in METHODS:
                teachers[t, m] = self.done(f'{t}-{m}-v1') or self.done(f'{t}-{m}-v0')
        fits = [job('fit', t, f'fit-{t}-{m}-s{s}', 'gpu', method=m, seed=s,
            teacher=str(teachers[t, m]), prepared=str(self.done('prepare-'+t)))
            for t in targets for m in METHODS for s in self.cfg['training_seeds']]
        self.batch(fits)
        qualifications, omitted = [], []
        for spec in fits:
            trained = self.done(spec['name']); result = read(trained/'result.json')
            teacher_ok = read(Path(spec['teacher'])/'result.json')['screen']['passed']
            order = result.get('qualified_checkpoint_order', [])
            if teacher_ok and order:
                qualifications.append(job('qualify', spec['target'], 'hmc-'+spec['name'], 'gpu',
                    seed=spec['seed'], method=spec['method'], prepared=spec['prepared'], training=str(trained), checkpoint=order[0]))
            else:
                omitted.append({'training': str(trained), 'reason': 'teacher probability screen failed' if not teacher_ok else 'no numerically valid map'})
        write(self.root/'qualification-plan.json', {'jobs': qualifications, 'unmet_prerequisites': omitted})
        self.sync('qualification_planned', [s['name'] for s in qualifications])
        self.batch(qualifications)
        summary = {'methods': {f'{t}/{m}': {'output': str(p), 'screen': read(p/'result.json')['screen']}
            for (t, m), p in teachers.items()}, 'training': [str(self.done(s['name'])) for s in fits],
            'qualifications': [str(self.done(s['name'])) for s in qualifications], 'omitted': omitted,
            'remaining': self.remaining(), 'method_ranking_established': False,
            'scientific_status': 'bounded tests completed; inspect each probability/training/HMC result separately'}
        write(self.root/'summary.json', summary)
        self.sync('completed', 'terminal result review and monograph result update')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'resume', 'status', 'check', 'worker'))
    parser.add_argument('--spec')
    args = parser.parse_args()
    if args.action == 'worker':
        return worker(args.spec)
    if args.action == 'status':
        print(json.dumps(read(CAMPAIGN/'next-phase.json') if (CAMPAIGN/'next-phase.json').exists() else {'status': 'not_started'}, indent=2)); return 0
    if args.action == 'check':
        env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='-1', TF_FORCE_GPU_ALLOW_GROWTH='true',
            TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2')
        return subprocess.call([sys.executable, '-m', 'pytest', '-q', 'tests/test_neutra_rare_regions.py'], cwd=LIVE_ROOT, env=env)
    with (SHARED/'master.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller = Controller()
        try:
            controller.run()
        except Exception as exc:
            controller.sync('repair_required', str(exc))
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
