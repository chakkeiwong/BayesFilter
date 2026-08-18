# Repo-Wide Tuning Reference Inventory

Generated 2026-08-16 with the following explicit text-matching command over
checked-out BayesFilter files. This inventory is reference coverage, not a
claim that every artifact is active code. The counts are provisional until the
command is rerun after source changes.

```bash
python - <<'PY'
from pathlib import Path
import re
roots=(Path('bayesfilter'),Path('tests'),Path('scripts'),Path('docs'))
pattern=re.compile(r'tuning|broad.?grid|fixed.?metric|kernel.?selection|kernel.?tuning|budget.?ladder', re.I)
for root in roots:
    files=[]
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in {'.py','.md','.json','.yaml','.yml','.toml'}:
            if pattern.search(path.read_text(errors='ignore')):
                files.append(path)
    print(root, len(files))
PY
```

**Matching files: 3553.**

| Root | Matching files | Python | Markdown | JSON/YAML/TOML |
|---|---:|---:|---:|---:|
| `bayesfilter` | 99 | 99 | 0 | 0 |
| `tests` | 126 | 126 | 0 | 0 |
| `scripts` | 14 | 14 | 0 | 0 |
| `docs` | 3314 | 161 | 1543 | 1610 |

Review routing: Python under `bayesfilter` is core/adjacent implementation; Python under `tests` and `scripts` is consumer/test code; Markdown and structured artifacts are provenance/history unless an active script reads them. A complete audit must sample active readers and classify the rest rather than treating text matches as APIs.

The full path list is intentionally regenerated with this command because the
repository contains thousands of historical benchmark/result documents.
