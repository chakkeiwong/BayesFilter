"""Diagnostic validation of completed original-author smoothing checkpoints."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
from bayesfilter.testing.zhao_cui_reference_utils import tree_fingerprint

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'third_party/audit/zhao_cui_tensor_ssm_p10/source'

def checkpoint_metadata(directory: Path, horizon: int):
    """A finished prefix remains usable when a later update times out.

    The source runner renames the MAT file atomically, then closes its summary
    row before resuming fitting. Live prefixes also verify the pinned source
    fingerprint; consumers still must check data, shapes and density parity.
    """
    manifest=json.loads((directory/'manifest.json').read_text())
    result_path=directory/'result.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else {'status':manifest['status']}
    status=result['status']
    if status not in ('complete','running','timeout','failed'):
        raise ValueError('invalid author fit status')
    if not (directory/f'smoothing-t{horizon:02d}.mat').exists():
        raise LookupError('smoothing checkpoint is not complete')
    if status=='running':
        expected=manifest.get('source_tree_sha256_before')
        if not expected or tree_fingerprint(SOURCE)!=expected:
            raise ValueError('live source fingerprint mismatch')
    elif not result.get('source_tree_unchanged'):
        raise ValueError('author source changed')
    if status!='complete':
        summary=directory/'smoothing-summary.csv'
        rows=list(csv.DictReader(summary.open())) if summary.exists() else []
        rows=[r for r in rows if int(r['time'])==horizon]
        if len(rows)!=1 or any(not math.isfinite(float(rows[0][key])) or float(rows[0][key])<=0 for key in ('ess','finite_fraction')):
            raise LookupError('smoothing checkpoint completion row is missing or invalid')
    return manifest,dict(result,parent_status=status,checkpoint_horizon=horizon)
