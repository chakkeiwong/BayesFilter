"""Diagnostic old/new seed accounting on saved charges; no numerical admission."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--tuning', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from bayesfilter.inference import hmc_acceptance_trials as current
from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionConfig

spec = importlib.util.spec_from_file_location('bayesfilter.inference._saved_trial_reference',
    args.reference/'bayesfilter/inference/hmc_acceptance_trials.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
execution = json.loads((args.tuning/'execution_spec.json').read_text())['execution']
result = json.loads((args.tuning/'tuning_checkpoint.json').read_text())['result']
groups = defaultdict(list)
for path in (args.tuning/'numerical_chunks').glob('*.json'):
    chunk = json.loads(path.read_text())
    # Only fields consumed by charge reconstruction; no trajectory is analyzed.
    groups[chunk['work']['work_item_id']].append({k: chunk[k] for k in
        ('trial_ordinal', 'trial_chunk_index', 'seed', 'runtime')})

def runtime():
    return SimpleNamespace(config=HMCCandidateExecutionConfig.from_payload(execution['config']),
        scope=SimpleNamespace(payload=lambda: execution['scope']), _evidence={},
        _partial=dict(groups), trial_seed_lineage=lambda seed: (tuple(seed),))

records = []
registries = []
for name, module in [('source15', reference), ('source16', current)]:
    binding = runtime()
    start = time.monotonic()
    module.initialize_seed_registry(binding, result)
    records.append(dict(implementation=name, wall_seconds=time.monotonic()-start,
        independent_streams=len(binding._trial_seed_registry)))
    registries.append(binding._trial_seed_registry)
assert registries[0] == registries[1]
payload = dict(status='identical_charge_registry', records=records,
    saved_chunks=sum(map(len, groups.values())), source=str(args.source),
    reference=str(args.reference), checkpoint=str(args.tuning/'tuning_checkpoint.json'),
    native_sampling=False, numerical_replay_authority=False, release_ready=False,
    timing_scope='single diagnostic timings, not a speed ranking')
args.output.write_text(json.dumps(payload, indent=2)+'\n')
print(json.dumps(payload))
