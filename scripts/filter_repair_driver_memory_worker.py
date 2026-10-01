"""Diagnostic fresh-process measurement of campaign history bookkeeping.

No TensorFlow target or GPU work is performed. The old driver is loaded only
as a frozen reference for its record loader and budget accounting.
"""

import argparse
import hashlib
import json
import resource
import subprocess
import threading
import time
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('VmRSS unavailable')


class FrozenHistory:
    """Ordinary path fixture: hide later campaign records from this comparison."""

    def __init__(self, index):
        self.index = index

    def glob(self, pattern):
        if pattern == 'run-*/run.json':
            return [Path(entry['path']) for entry in self.index['records']]
        if pattern == 'supplemental-compute-*.json':
            return [Path(entry['path']) for entry in self.index['supplemental']]
        raise ValueError(pattern)


def main(args):
    index_bytes = args.index.read_bytes()
    index = json.loads(index_bytes)
    assert index['through_run'] == 3870 and index['baseline'] == '4c37f9f40'
    for entry in index['records'] + index['supplemental']:
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
    latest = json.loads(Path(index['records'][-1]['path']).read_text())
    key, hashes = latest['key'], latest['source_sha256']
    if args.arm == 'prior':
        source = subprocess.check_output(['git', 'show', '4c37f9f40:scripts/run_filter_repair_campaign.py'],
            cwd=ROOT, text=True)
    else:
        source = (ROOT / 'scripts/run_filter_repair_campaign.py').read_text()
    driver = ModuleType('_driver_history_reference')
    driver.__file__ = str(ROOT / 'scripts/run_filter_repair_campaign.py')
    exec(compile(source, driver.__file__, 'exec'), driver.__dict__)  # noqa: S102 -- fixed diagnostic authority
    driver.OUTPUT = FrozenHistory(index)
    before = rss_bytes()
    lifetime_peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    samples = [before]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(.01):
            samples.append(rss_bytes())

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    tick = time.monotonic()
    if args.arm == 'prior':
        rows = driver.records()
        attempts = len([row for row in rows if row['key'] == key and row['source_sha256'] == hashes])
    else:
        rows, attempts = driver.history_summary(driver.records(), key, hashes)
    charges = {device: driver.charged_seconds(rows, device) for device in ('CPU', 'GPU')}
    seconds = time.monotonic() - tick
    retained = rss_bytes()
    stop.set()
    sampler.join(timeout=2)
    samples.append(retained)
    output = {'schema': 'filter_repair_driver_history_memory.v1', 'arm': args.arm,
        'baseline': index['baseline'], 'history_sha256': hashlib.sha256(index_bytes).hexdigest(),
        'driver_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'run_count': len(rows), 'next_run_number': len(rows) + 1,
        'attempt_key': key, 'exact_source_attempts': attempts, 'charged_seconds': charges,
        'seconds': seconds, 'rss_before_bytes': before, 'rss_retained_bytes': retained,
        'sampled_peak_rss_bytes': max(samples), 'rss_sample_count': len(samples),
        'lifetime_peak_before_bytes': lifetime_peak_before,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'scope': 'CPU host record loading only; no TensorFlow compiler or tensor allocation claim'}
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(output, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('prior', 'streamed'), required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
