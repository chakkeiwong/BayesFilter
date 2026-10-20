"""CPU-only observer input check; no service mutation or sampler work."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_hmc_observer',root/'observer.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
request=module.read(root/'request.json')
assert module.sha(root/'observer.py')==request['observer_sha256']
assert module.sha(root/'test_monitor.py')==request['tests_sha256']
assert module.sha(Path(request['repo'])/'scripts/monitor_hmc_v7_confirmation.py')==request['observer_sha256']
assert module.sha(Path(request['repo'])/'tests/test_hmc_confirmation_monitor.py')==request['tests_sha256']
started=time.monotonic()
snapshot=module.inspect_run(request,dict(LoadState='loaded',ActiveState='active',MainPID='2128000'))
elapsed=time.monotonic()-started
assert snapshot['status']=='running' and not snapshot['disk_stop_required']
assert snapshot['original_denominator']==96 and not snapshot['release_ready']
assert snapshot['complete_delivery_count']>=2
assert {'r01-nonlinear','r01-funnel_residual'}<=set(snapshot['resource_deferred_slots'])
for slot in snapshot['resource_deferred_slots']:
    assert not module.recovery_eligible(request,slot,primary_active=True,tried=set(),remaining_seconds=189073)
result=dict(status='passed',timestamp_utc=datetime.now(timezone.utc).isoformat(),
    inspect_wall_seconds=elapsed,live_snapshot=snapshot,source_and_test_hashes_checked=True,
    original_campaign_untouched=True,review='PASS: real completed-member/seed/device records validate; live-primary recovery refused; original outcomes remain authoritative; monitoring is observational except declared evidence/storage vetoes.',
    numerical_sampling=False,release_ready=False)
module.write(root/'live-input-audit.json',result)
print(json.dumps({k:result[k] for k in ('status','inspect_wall_seconds','original_campaign_untouched')}))
print(json.dumps({k:snapshot[k] for k in ('outcomes_recorded','complete_delivery_count','resource_deferred_slots','active_slot','free_disk_bytes')}))
