#!/usr/bin/env python3
"""Bounded local reference scheduler; no scientific selection or model changes.

Observes existing campaign manifests, runs a declared queue at no more than two
full jobs, and counts aggregate elapsed job time including failed attempts.
Only processes launched by this supervisor can be terminated by it.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = {
    'docs/benchmarks/run_zhao_cui_publication_replication.py',
    'docs/benchmarks/run_zhao_cui_quadratic_score_reference.py',
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inspect_budget(campaign, exclusions):
    now = dt.datetime.now(dt.timezone.utc)
    total, running = 0.0, set()
    for path in campaign.glob('*/manifest.json'):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            raise RuntimeError(f'manifest being written or invalid: {path}')
        if path.parent.name in exclusions:
            continue
        elapsed = float(record.get('wall_seconds', 0.0))
        if record.get('status') == 'running':
            elapsed = (now - dt.datetime.fromisoformat(record['started_utc'])).total_seconds()
            running.add(str(path.parent.resolve()))
        total += elapsed
    return total, running


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    queue_path = args.queue.resolve()
    queue = json.loads(queue_path.read_text())
    campaign = (ROOT / queue['campaign_root']).resolve()
    budget = float(queue['additional_budget_seconds'])
    limit = int(queue['maximum_concurrent_full_jobs'])
    if limit not in (1, 2) or not 0 < budget <= 172800:
        raise ValueError('outside declared resource limits')
    exclusions = set(queue['prior_run_exclusions'])
    entries = queue['entries']
    paths = []
    for entry in entries:
        command = entry['command']
        path = (ROOT / entry['output_directory']).resolve()
        if len(command) < 2 or command[0] != 'python3' or command[1] not in RUNNERS or path.parent != campaign or path.exists():
            raise ValueError(f'invalid command or reused output directory: {entry["name"]}')
        if command[command.index('--output-root') + 1] != entry['output_directory']:
            raise ValueError('command output path differs from queue')
        paths.append(str(path))
    if len(set(paths)) != len(paths):
        raise ValueError('duplicate output directories')
    consumed, running = inspect_budget(campaign, exclusions)
    if args.dry_run:
        print(json.dumps(dict(status='dry_run',running_full_jobs=len(running),
                             consumed_hours=consumed / 3600, remaining_hours=(budget-consumed) / 3600,
                             queued=[entry['name'] for entry in entries])))
        return
    status_path = queue_path.with_suffix('.status.json')
    if status_path.exists():
        raise ValueError('supervisor output already exists; use a new queue')
    owned, launched, next_index = {}, [], 0
    state = dict(status='running', queue=str(queue_path), supervisor_pid=os.getpid(), launches=launched)
    while True:
        try:
            consumed, running = inspect_budget(campaign, exclusions)
        except RuntimeError:
            time.sleep(1)
            continue
        for path, item in list(owned.items()):
            code = item['process'].poll()
            if code is None:
                running.add(path)
                continue
            item['log'].close()
            item['record'].update(returncode=code, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            manifest_path = Path(path) / 'manifest.json'
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest.get('status') == 'running':
                    manifest.update(status='supervisor_observed_incomplete_exit',
                                    wall_seconds=time.monotonic()-item['start'],
                                    supervisor_returncode=code)
                    save(manifest_path, manifest)
            del owned[path]
            running.discard(path)
        state.update(recorded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                     consumed_seconds=consumed, remaining_seconds=max(0, budget-consumed),
                     running_directories=sorted(running), next_queue_index=next_index)
        save(status_path, state)
        if consumed >= budget:
            for item in owned.values():
                os.killpg(item['process'].pid, signal.SIGTERM)
            time.sleep(2)
            for path, item in owned.items():
                if item['process'].poll() is None:
                    os.killpg(item['process'].pid, signal.SIGKILL)
                item['log'].close()
                manifest_path = Path(path) / 'manifest.json'
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    manifest.update(status='campaign_budget_exhausted', wall_seconds=time.monotonic()-item['start'])
                    save(manifest_path, manifest)
            state['status'] = 'campaign_budget_exhausted'
            save(status_path, state)
            break
        while next_index < len(entries) and len(running) < limit:
            entry = entries[next_index]
            dependencies_ready = True
            for name in entry.get('requires_completed_runs', []):
                dependency = campaign / name / 'manifest.json'
                if dependency.parent.parent != campaign:
                    raise ValueError('dependency must be a campaign run name')
                if not dependency.exists():
                    dependencies_ready = False
                    break
                try:
                    dependency_status = json.loads(dependency.read_text()).get('status')
                except json.JSONDecodeError:
                    dependencies_ready = False
                    break
                if dependency_status == 'running':
                    dependencies_ready = False
                    break
                if dependency_status != 'complete':
                    state.update(status='dependency_failed', dependency=str(dependency),
                                 dependency_status=dependency_status)
                    save(status_path, state)
                    print(json.dumps(state))
                    return
            if not dependencies_ready:
                break
            log = (campaign / (entry['name'] + '-launch.log')).open('x')
            start = time.monotonic()
            process = subprocess.Popen(entry['command'], cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record = dict(name=entry['name'], command=entry['command'], pid=process.pid,
                          started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            launched.append(record)
            path = paths[next_index]
            owned[path] = dict(process=process, log=log, start=start, record=record)
            running.add(path)
            next_index += 1
            state['next_queue_index'] = next_index
            save(status_path, state)
        if next_index == len(entries) and not owned:
            state['status'] = 'queue_finished'
            save(status_path, state)
            break
        time.sleep(10)
    print(json.dumps(dict(status=state['status'], status_path=str(status_path))))


if __name__ == '__main__':
    main()
