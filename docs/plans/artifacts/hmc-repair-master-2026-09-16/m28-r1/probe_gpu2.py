"""Trusted bounded GPU 2 readiness; installed generic probe supports only 0/1."""
import json
from pathlib import Path
from bayesfilter.testing.inference_validation.designs import ValidationDesign
from bayesfilter.testing.inference_validation.execution import configure_worker
root=Path(__file__).resolve().parent
suite=json.loads((root/'designs-r1/m28-exact-2026092381.json').read_text())
runtime=configure_worker(ValidationDesign.from_payload(suite['designs'][0]))
assert runtime['visible_devices']=='2'
(root/'gpu2-readiness.json').write_text(json.dumps(runtime,indent=2)+'\n')
print(json.dumps(runtime))
