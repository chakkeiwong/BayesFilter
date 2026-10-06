#!/usr/bin/env python3
"""Resume the attribution queue after the current pilot controller exits.

This is a bounded local process supervisor, not an AI repair agent. It only
launches the fixed attribution command and stops on numerical/worker failures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/ubuntu/python/BayesFilter')
CAMPAIGN=ROOT/'docs/plans/artifacts/neutra-source-fit-remedy-2026-10-04/campaign-r1'
REQUEST=CAMPAIGN/'attribution-supervision-request.json'
STATUS=CAMPAIGN/'attribution-supervisor.json'


def write(value):
    temp=STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(STATUS)


def main():
    request=json.loads(REQUEST.read_text())
    pid=int(request['previous_controller_pid'])
    status=dict(status='waiting_for_pilot_controller',previous_controller_pid=pid,
        supervisor_pid=os.getpid(),started_unix=time.time(),command=[
        'bash',str(ROOT/'scripts/run_neutra_scientific_campaign.sh'),'attribution'])
    write(status)
    while True:
        try:
            command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ')
            process=Path(f'/proc/{pid}/status').read_text()
        except FileNotFoundError:break
        if b'run_neutra_scientific_campaign_master.py attribution' not in command or '\nState:\tZ' in process:
            break
        time.sleep(15)
    state=json.loads((CAMPAIGN/'attribution-state.json').read_text())
    current=json.loads((CAMPAIGN/'state.json').read_text())
    last=current['attempts'][-1]
    if current.get('active') or last['status']!='complete' or state['status'] in ('repair_required','control_failed'):
        status.update(status='stopped_for_inspection',reason='active/unsettled worker or invalid prior attempt')
        write(status)
        return 1
    status.update(status='resuming_reviewed_queue',resumed_unix=time.time())
    write(status)
    with (CAMPAIGN/'attribution-supervisor.log').open('a') as log:
        code=subprocess.call(status['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    status.update(status='complete' if code==0 else 'stopped_for_inspection',exit_code=code,
                  finished_unix=time.time())
    write(status)
    return code


if __name__=='__main__':
    raise SystemExit(main())
