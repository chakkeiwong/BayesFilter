"""One bounded CPU/XLA reference price under the release plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
source = args.source.resolve()
records = json.loads((source.parent / 'source-manifest.json').read_text())
for name, expected in records.items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
sys.path.insert(0, str(source))
os.chdir(source)
from bayesfilter.testing.acceptance_release_validation import full_search_configuration
from bayesfilter.testing.acceptance_decision_models import run_model

config = full_search_configuration('lgssm_qr', seed=(20261002, 2501), wall_seconds=900)
manifest = dict(source=str(source),
    source_manifest_sha256=hashlib.sha256((source.parent / 'source-manifest.json').read_bytes()).hexdigest(),
    command=sys.argv, environment=sys.executable,
    git_commit=json.loads((source.parent / 'assembly.json').read_text())['git_commit'],
    seed=config['seed'], data=config['data'], configuration=config,
    gpu_intentionally_hidden=True, jit_compile=True,
    plan='docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md',
    scope='CPU reference cost diagnosis; no GPU/default/multi-seed delivery claim')
result = run_model(config, args.output.resolve(), manifest=manifest)
print(json.dumps({k: result[k] for k in ('completion_status', 'verified_candidate_ids',
                                       'wall_seconds', 'evidence_accounting')}))
