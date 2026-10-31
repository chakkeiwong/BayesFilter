"""Preserve raw diagnostic evidence; export standards-compliant equivalents."""
import hashlib
import json
from pathlib import Path
from strict_json import strict_numbers

campaign = Path(__file__).resolve().parent
exports = campaign / 'strict-json'
exports.mkdir(exist_ok=False)
index = []
for source in sorted(campaign.rglob('*.json')):
    if exports in source.parents or 'data' in source.parts:
        continue
    raw = source.read_text()
    special = []
    def constant(value):
        special.append(value)
        return {'__nonfinite_float__': value}
    decoded = json.loads(raw, parse_constant=constant)
    if not special:
        continue
    relative = source.relative_to(campaign)
    destination = exports / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(decoded, indent=2, allow_nan=False) + '\n')
    index.append(dict(original=str(relative), export=str(destination.relative_to(campaign)),
                      original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      export_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                      tagged_nonfinite_count=len(special)))
(exports / 'index.json').write_text(json.dumps(dict(
    encoding={'__nonfinite_float__': 'NaN | Infinity | -Infinity'},
    purpose='Lossless JSON-compatible representation of raw diagnostics; originals preserved.',
    exports=index), indent=2, allow_nan=False) + '\n')
print(json.dumps({'strict_exports':len(index), 'nonfinite_values':sum(r['tagged_nonfinite_count'] for r in index)}))
