"""Close only the audited, completed M21 worker after a final teardown grace."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time

root = Path(__file__).resolve().parent
audit_path = root/'public-final-audit-r2/result.json'
audit = json.loads(audit_path.read_text())
key = 'm21-merged-confirmation-beta_binomial'
checked = audit['terminal_public_groups'][key]
assert checked['complete_inventory'] and checked['completed'] == checked['planned'] == 128
assert checked['complete_final_result_written']
job = root.parent/'m21-r2/public-confirmation-cpu-r1/beta_binomial'/key
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(job/'assessment.json') == checked['assessment_sha256']
paths = [job/name for name in ('assessment.json', 'attempt-001-result.json', 'result.json')]
hashes = {str(p): digest(p) for p in paths}
pid = 3899133
proc = Path(f'/proc/{pid}')
expected = [str(job/'design.json'), str(job), '35000', '1']
def identity():
    try:
        cmd = (proc/'cmdline').read_bytes().decode().split('\0')[:-1]
        stat = (proc/'stat').read_text().split()
    except FileNotFoundError:
        return None
    assert '_worker' in cmd and cmd[-4:] == expected, 'worker PID identity changed'
    return stat[21]
token = identity()
record = {'pid': pid, 'expected_job': str(job), 'audit_sha256': digest(audit_path),
    'final_evidence_sha256': hashes, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'grace_seconds': 60, 'role': 'post-output resource cleanup, no sample or evidence modification',
    'worker_process_exit_must_remain_recorded': True}
start = time.monotonic()
if token is not None:
    for _ in range(60):
        if identity() is None:
            break
        time.sleep(1)
now = identity()
assert all(digest(Path(p)) == h for p, h in hashes.items()), 'final evidence changed during grace'
if now is not None:
    assert now == token
    os.kill(pid, signal.SIGTERM)
    record['action'] = 'SIGTERM_after_verified_complete_output_and_grace'
else:
    record['action'] = 'natural_exit_no_signal'
record['elapsed_seconds'] = time.monotonic()-start
record['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
with (root/'completed-worker-cleanup.json').open('x') as f:
    json.dump(record, f, indent=2)
    f.write('\n')
print(json.dumps(record), flush=True)
