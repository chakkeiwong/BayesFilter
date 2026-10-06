#!/usr/bin/env python3
"""Apply the September 30 allocation after the active master releases its lock.

No numerical imports or changes to scientific configuration. The additional
48 compute-hours are split as in the previous allocation: CPU 32, GPU 16.
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'docs/plans/artifacts/neutra-warm-start-master-2026-09-29/campaign-r1'
ALLOCATION = 'budget-extension-20260930-r1'


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def apply_allocation(root):
    """Called with master.lock held; absolute ceilings make retries idempotent."""
    directory = root / ALLOCATION
    allocation = load(directory / 'allocation.json')
    config = load(root / 'config.json')
    before = allocation['previous_ceilings']
    after = allocation['new_ceilings']
    current = {key: config[key] for key in before}
    if current not in (before, after):
        raise ValueError('Resource ceilings changed since allocation; reconcile recorded costs and funding')
    if current == before:
        config.update(after)
        config['authorization'] = allocation['authorization']
        write(root / 'config.json', config)
    state = load(root / 'state.json')
    repair = load(root / 'repair-master-state.json')
    last_decision = next((item.get('decision') for item in reversed(repair.get('decisions', []))), None)
    budget_stop = state['status'] == 'budget_exhausted' and (
        repair['status'] == 'budget_exhausted' or (
            repair['status'] == 'stopped_with_preserved_evidence' and last_decision == 'budget_exhausted'))
    allocation.update(applied_unix=time.time(),
                      status='applied', prior_master_status=state['status'],
                      automatic_resume=budget_stop)
    write(directory / 'allocation.json', allocation)
    return allocation['automatic_resume']


def run(root=CAMPAIGN, *, resume=None):
    root = Path(root)
    directory = root / ALLOCATION
    status_path = directory / 'handover-state.json'
    with (directory / 'handover.lock').open('a') as handover_lock:
        fcntl.flock(handover_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {'status': 'waiting_for_active_master', 'pid': os.getpid(),
                  'started_unix': time.time(), 'allocation': str(directory / 'allocation.json')}
        write(status_path, status)
        with (root / 'master.lock').open('a') as master_lock:
            fcntl.flock(master_lock, fcntl.LOCK_EX)
            try:
                should_resume = apply_allocation(root)
                status.update(status='resuming_after_budget_exhaustion' if should_resume
                              else 'allocation_applied_no_resume_required', updated_unix=time.time())
                write(status_path, status)
                if should_resume:
                    if resume is None:
                        spec = importlib.util.spec_from_file_location(
                            'repair_budget_continuation', ROOT / 'scripts/run_neutra_warm_start_repair_master.py')
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        controller = module.Controller(root)
                        controller.state['status'] = 'ready'
                        controller.record['budget_basis'] = load(directory / 'allocation.json')['authorization']
                        controller.refresh('budget_extension_applied',
                                           decision='resume pending work using the recorded additional 48 compute-hours',
                                           evidence=str(directory / 'allocation.json'))
                        try:
                            controller.run()
                        except BaseException as exc:
                            controller.refresh('stopped_with_preserved_evidence', decision=str(exc))
                            controller.write_results()
                            raise
                    else:
                        resume(root)
                    status.update(status='resumed_controller_returned', updated_unix=time.time())
                    write(status_path, status)
            except BaseException as exc:
                status.update(status='requires_inspection', error=repr(exc), updated_unix=time.time())
                write(status_path, status)
                raise


if __name__ == '__main__':
    run()
