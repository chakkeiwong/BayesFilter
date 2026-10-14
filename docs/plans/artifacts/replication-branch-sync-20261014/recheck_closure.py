import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
checkout = Path(sys.argv[1]).resolve()
archive = Path('/home/chakwong/BayesFilter')
audit_path = checkout / 'docs/benchmarks/audit_kdm_source_coverage.py'
spec = importlib.util.spec_from_file_location('recovered_source_audit', audit_path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.ROOT = checkout
audit.BASE = archive / 'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
audit.AUTHOR = archive / '.localresources/code/younis-mdpf-neurips-2023'
audit.PAPER = str(archive / audit.PAPER)
output = checkout / sys.argv[2]
status = audit.run(SimpleNamespace(output=str(output)))
binding = dict(
    schema='migrated_runtime_archived_evidence_binding_v1',
    current_code_root=str(checkout),
    archived_evidence_root=str(audit.BASE),
    pinned_author_root=str(audit.AUTHOR),
    paper_path=audit.PAPER,
    command=[sys.executable, '-B', *sys.argv],
    cpu_only=True,
    CUDA_VISIBLE_DEVICES='-1',
    numerical_execution=False,
    runtime_checked_by_existing_auditor=True,
    exit_code=status,
)
(output / 'archive-binding.json').write_text(json.dumps(binding, indent=2) + '\n')
sys.exit(status)
