"""Allocation-only diagnostic of Linux memory counters; never a runtime kernel."""

import argparse
import hashlib
import json
import mmap
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def snapshot():
    began = time.perf_counter()
    status, smaps = {}, {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem'):
            status[key] = int(value.split()[0])*1024
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('Rss', 'Pss', 'Anonymous', 'AnonHugePages'):
            smaps[key] = int(value.split()[0])*1024
    return {'status': status, 'smaps_rollup': smaps,
        'rusage_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'read_order': 'status, smaps_rollup, rusage; not atomic',
        'read_wall_seconds': time.perf_counter()-began}


def child(mode, output):
    original = sorted(os.sched_getaffinity(0))
    cpus = original[:1] if mode == 'pinned' else original[:32]
    assert mode != 'spread' or len(cpus) == 32, 'requires32 available CPUs for the declared contrast'
    size, count, page = 17*1024**2, 32, os.sysconf('SC_PAGE_SIZE')
    assert size % page == 0
    mappings, rows = [], []
    start = time.perf_counter()
    before = snapshot()
    try:
        for index in range(count):
            cpu = cpus[index % len(cpus)]
            os.sched_setaffinity(0, {cpu})
            assert os.sched_getaffinity(0) == {cpu}
            allocation = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                                   prot=mmap.PROT_READ | mmap.PROT_WRITE)
            mappings.append(allocation)
            for offset in range(0, size, page):
                allocation[offset] = 1
            rows.append({'segment': index, 'cpu': cpu, 'touched_bytes': size*(index+1),
                         'memory': snapshot()})
        assert all(allocation[0] == allocation[-page] == 1 for allocation in mappings)
    finally:
        for allocation in mappings:
            allocation.close()
        os.sched_setaffinity(0, original)
    after_release = snapshot()
    forbidden = [name for name in sys.modules if name.split('.')[0] in ('tensorflow', 'numpy', 'jax', 'torch')]
    growth = rows[-1]['memory']['smaps_rollup']['Rss']-before['smaps_rollup']['Rss']
    result = {'schema': 'filter_repair_os_memory_allocation.v1', 'mode': mode,
        'original_affinity': original, 'allocation_cpus': cpus, 'page_size': page,
        'segment_bytes': size, 'segment_count': count, 'touched_bytes': size*count,
        'before': before, 'allocations': rows, 'after_release': after_release,
        'observed_smaps_growth_bytes': growth, 'touched_growth_error_bytes': growth-size*count,
        'forbidden_imports': forbidden, 'pid': os.getpid(), 'python': sys.version,
        'kernel': platform.release(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'elapsed_seconds': time.perf_counter()-start,
        'passed': len(rows) == count and not forbidden and abs(growth-size*count) <= 2*1024**2}
    output.write_text(json.dumps(result, indent=2)+'\n')
    assert result['passed'], 'allocation diagnostic contract failed; preserved result'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--child', choices=('pinned', 'spread'))
    args = parser.parse_args()
    if args.child:
        child(args.child, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    plan = 'docs/plans/filter_gradient_os_memory_accounting_20260929.md'
    receipt = args.output.parent/'supplemental-compute-os-memory-20260929-r1.json'
    charge = {'schema': 'filter_repair_supplemental_compute.v1', 'device': 'CPU',
        'charged_seconds': 120., 'measurement_kind': 'reserved_timeout',
        'reason': 'Two sequential allocation-only OS counter diagnostics, no TensorFlow or GPU.',
        'output': str(args.output), 'plan': plan, 'state': 'running'}
    with receipt.open('x') as stream:
        json.dump(charge, stream, indent=2)
        stream.write('\n')
    start = time.perf_counter()
    manifest = {'schema': 'filter_repair_os_memory_driver.v1', 'plan': plan,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
        'cwd': os.getcwd(), 'python': sys.version, 'kernel': platform.release(),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'hardware': {'cpu_count': os.cpu_count(), 'affinity': sorted(os.sched_getaffinity(0))},
        'children': [], 'headers': {}, 'state': 'running'}
    (args.output/source.name).write_bytes(source.read_bytes())
    header_root = Path('/usr/src')/('linux-hwe-6.8-headers-'+platform.release().removesuffix('-generic'))/'include/linux'
    for name, first, last in (('mm.h', 2656, 2749), ('percpu_counter.h', 80, 125)):
        path = header_root/name
        if path.exists():
            body = path.read_bytes()
            extract = '\n'.join(f'{i}: {line}' for i, line in enumerate(body.decode().splitlines(), 1)
                                if first <= i <= last)+'\n'
            (args.output/(name+'.excerpt.txt')).write_text(extract)
            manifest['headers'][name] = {'path': str(path), 'sha256': hashlib.sha256(body).hexdigest(),
                'excerpt_sha256': hashlib.sha256(extract.encode()).hexdigest()}
    try:
        for mode in ('pinned', 'spread'):
            command = [sys.executable, '-I', str(source), '--child', mode,
                       '--output', str(args.output/(mode+'.json'))]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}
            remaining = 120-(time.perf_counter()-start)
            assert remaining > 0
            with (args.output/(mode+'.log')).open('w') as log:
                run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=min(60, remaining), check=False)
            manifest['children'].append({'mode': mode, 'command': command,
                                         'returncode': run.returncode, 'cuda_visible_devices': '-1'})
            assert run.returncode == 0, (mode, run.returncode)
        records = {mode: json.loads((args.output/(mode+'.json')).read_text()) for mode in ('pinned', 'spread')}
        summary = {}
        for mode, result in records.items():
            memory = result['allocations'][-1]['memory']
            summary[mode] = {'final_status_rss_bytes': memory['status']['VmRSS'],
                'final_smaps_rss_bytes': memory['smaps_rollup']['Rss'],
                'final_rusage_peak_bytes': memory['rusage_maxrss_bytes'],
                'final_rusage_deficit_bytes': memory['smaps_rollup']['Rss']-memory['rusage_maxrss_bytes'],
                'touched_growth_error_bytes': result['touched_growth_error_bytes'], 'passed': result['passed']}
        manifest.update(summary=summary, state='passed')
    finally:
        elapsed = time.perf_counter()-start
        if manifest['state'] == 'running':
            manifest['state'] = 'failed'
        manifest['elapsed_seconds'] = elapsed
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        charge.update(charged_seconds=elapsed, measurement_kind='driver_wall_seconds', state=manifest['state'])
        receipt.write_text(json.dumps(charge, indent=2)+'\n')
    print(json.dumps({'state': manifest['state'], 'elapsed_seconds': elapsed, 'summary': manifest.get('summary')}))


if __name__ == '__main__':
    main()
