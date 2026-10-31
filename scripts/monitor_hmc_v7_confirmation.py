"""Observe a frozen confirmation; retry only unsampled resource deferrals.

Primary outcomes are never modified. This observer cannot declare a release.
It uses only the standard library and never imports an accelerator framework.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def service_status(unit):
    properties = ('LoadState', 'ActiveState', 'SubState', 'MainPID', 'Result',
                  'ExecMainStatus', 'ExecMainStartTimestampMonotonic',
                  'ExecMainExitTimestampMonotonic', 'ControlGroup')
    process = subprocess.run(['systemctl', '--user', 'show', unit,
        *['--property=' + name for name in properties]], capture_output=True,
        text=True, timeout=10)
    result = dict(line.split('=', 1) for line in process.stdout.splitlines() if '=' in line)
    if process.returncode and result.get('LoadState') != 'not-found':
        raise RuntimeError('service inspection failed: ' + process.stderr[-1000:])
    if result.get('LoadState') != 'not-found' and 'ActiveState' not in result:
        raise ValueError('incomplete service inspection')
    return result


def service_active(service):
    return service.get('ActiveState') in {'activating', 'active', 'reloading', 'deactivating'}


def inspect_run(request, service, *, now_epoch=None, free_bytes=None):
    """Check small issued/terminal records, not a replacement for raw audit."""
    root, source = Path(request['run_root']), Path(request['source'])
    design = read(request['design'])
    if sha(request['design']) != request['design_sha256'] or read(root/'design.json') != design:
        raise ValueError('frozen confirmation design changed')
    manifest_path = source.parent/'source-manifest.json'
    if sha(manifest_path) != design['source_manifest_sha256']:
        raise ValueError('source manifest changed')
    for name, expected in read(manifest_path).items():
        if sha(source/name) != expected:
            raise ValueError('frozen source file changed: ' + name)
    if sha(root/'runner.py') != request['runner_sha256']:
        raise ValueError('frozen worker changed')
    planned = {slot['slot_id']: slot for slot in design['slots']}
    progress_path = root/'progress.json'
    rows = read(progress_path).get('outcomes', []) if progress_path.exists() else []
    if len(rows) > len(planned):
        raise ValueError('too many primary outcomes')
    successful, deferrals, failures = [], [], []
    for expected, row in zip(design['slots'], rows):
        sid = expected['slot_id']
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError('primary outcome differs from frozen slot/order')
        if row.get('source_manifest_sha256') != design['source_manifest_sha256']:
            raise ValueError('primary outcome source changed')
        code = row.get('exit_code')
        if type(code) is not int or code not in {0, 1, 2, 3, 124}:
            raise ValueError('unknown primary exit code')
        if code == 0:
            directory = root/sid
            cfg = read(directory/'configuration.json')
            model = read(directory/'model/result.json')
            manifest = read(directory/'manifest.json')
            frozen_cfg = read(Path(request['design']).parent/(sid+'-config.json'))
            if cfg != frozen_cfg or model.get('sampling_streams') != expected['seed']:
                raise ValueError('completed slot configuration or seed differs')
            if (manifest.get('configuration_sha256') != sha(directory/'configuration.json')
                    or manifest.get('source_manifest_sha256') != design['source_manifest_sha256']
                    or manifest.get('runner_sha256') != request['runner_sha256']
                    or manifest.get('gpu_uuid') != design['gpu_uuid']):
                raise ValueError('completed slot provenance differs')
            memory = manifest.get('memory_policy', {})
            if (memory.get('all_physical_devices_memory_growth') is not True
                    or memory.get('configured_before_logical_device_initialization') is not True
                    or manifest.get('jit_compile') is not True):
                raise ValueError('completed slot memory/XLA policy invalid')
            verified = model.get('verified_candidate_ids', [])
            expected_ids = {cid for cid, state in model.get('candidate_states', {}).items() if state == 'verified'}
            members = [read(p)['candidate_id'] for p in (directory/'model').glob('*-member.json')]
            device = read(directory/'device_result.json')
            if (model.get('completion_status') != 'complete'
                    or model.get('checkpoint_recomputed') is not True
                    or model.get('expectation_met') is not True
                    or model.get('classification') != 'confirmation'
                    or not verified or len(verified) != len(set(verified))
                    or len(members) != len(set(members))
                    or set(verified) != expected_ids or set(members) != expected_ids
                    or not device.get('sample_devices')
                    or any('GPU:0' not in value for value in device['sample_devices'])):
                raise ValueError('exit-zero slot lacks complete all-member GPU evidence')
            successful.append(sid)
        elif code == 3:
            deferrals.append(sid)
        else:
            failures.append(dict(slot_id=sid, exit_code=code, disposition=row.get('disposition')))
    now_epoch = time.time() if now_epoch is None else now_epoch
    active = None
    if service_active(service) and len(rows) < len(design['slots']):
        sid = design['slots'][len(rows)]['slot_id']
        directory = root/sid
        artifacts = [directory/name for name in ('resource.json', 'manifest.json',
            'model/stage_timing.json', 'model/tuning/tuning_checkpoint.json', 'model/result.json')]
        changed = [p.stat().st_mtime for p in artifacts if p.exists()]
        age = now_epoch-max(changed) if changed else None
        active = dict(slot_id=sid, progress_age_seconds=age,
            warning=age is not None and age > request['stale_warning_seconds'])
    terminal_path = root/'result.json'
    terminal = read(terminal_path) if terminal_path.exists() else None
    if terminal is not None and terminal.get('outcomes') != rows:
        raise ValueError('terminal outcomes differ from primary progress')
    if terminal is not None and (terminal.get('original_denominator') != len(planned)
                                  or len(rows) != len(planned)):
        raise ValueError('terminal denominator changed')
    free = shutil.disk_usage(root).free if free_bytes is None else free_bytes
    footprints=request.get('allocated_bytes_by_family')
    projected=None
    if footprints:
        outstanding=design['slots'][len(rows):]+[planned[sid] for sid in deferrals]
        projected=sum(footprints[slot['family']] for slot in outstanding)
    status = ('running' if service_active(service) else
              'finished_pending_audit' if terminal and terminal.get('status') == 'execution_complete'
              else 'error_requires_diagnosis')
    return dict(status=status, service=service, original_denominator=len(planned),
        outcomes_recorded=len(rows), complete_delivery_count=len(successful),
        successful_slots=successful, resource_deferred_slots=deferrals,
        other_failures=failures, active_slot=active, free_disk_bytes=free,
        projected_remaining_allocated_bytes=projected,
        storage_projection_warning=projected is not None and free < 1.2*projected,
        disk_stop_required=free < request['disk_stop_floor_bytes'],
        terminal_status=terminal.get('status') if terminal else None,
        terminal_delivery=terminal.get('delivery') if terminal else None,
        original_outcomes_unchanged=True, release_ready=False, default_promoted=False)


def recovery_eligible(request, slot_id, *, primary_active, tried, remaining_seconds):
    """A failed sampled candidate must never become a resource retry."""
    if primary_active or slot_id in tried:
        return False
    design = read(request['design'])
    process_cap = (design['search_cap_seconds'] + design['closeout_cap_seconds']
                   + design['readiness']['wait_cap_seconds'])
    if remaining_seconds < process_cap + request['cleanup_seconds']:
        return False
    root = Path(request['run_root'])
    terminal = read(root/'result.json') if (root/'result.json').exists() else None
    if not terminal or terminal.get('status') != 'execution_complete':
        return False
    planned = next((s for s in design['slots'] if s['slot_id'] == slot_id), None)
    rows = [r for r in terminal.get('outcomes', []) if r.get('slot_id') == slot_id]
    directory = root/slot_id
    if not planned or len(rows) != 1 or rows[0].get('exit_code') != 3:
        return False
    if any(rows[0].get(key) != value for key, value in planned.items()):
        raise ValueError('deferred primary slot identity changed')
    if (directory/'model').exists() or not (directory/'resource.json').exists():
        return False
    resource = read(directory/'resource.json')
    if (resource.get('gpu_initialized') is not False or resource.get('ready') is not False
            or resource.get('disposition') != 'resource_deferred'):
        return False
    cfg = read(directory/'configuration.json')
    frozen = read(Path(request['design']).parent/(slot_id+'-config.json'))
    if cfg != frozen or cfg.get('seed') != planned['seed']:
        raise ValueError('resource retry configuration differs from original')
    return True


def gpu_snapshot(gpu_uuid):
    rows = subprocess.check_output(['nvidia-smi',
        '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True, timeout=5).splitlines()
    selected = [row for row in rows if row.split(',')[0].strip() == gpu_uuid]
    if len(selected) != 1:
        raise RuntimeError('selected GPU is missing or duplicated in readiness inventory')
    return dict(inventory=selected[0], interpretation='Resource snapshot only; low utilization is not a sampler failure.')


def primary_charge(request, service, terminal, *, now_monotonic):
    """Actual exit timestamps when available; otherwise a conservative bound."""
    start = request['primary_start_monotonic_seconds']
    exit_us = int(service.get('ExecMainExitTimestampMonotonic', '0'))
    reported = float(terminal.get('wall_seconds', 0)) if terminal else 0.
    if not math.isfinite(reported) or reported < 0:
        raise ValueError('invalid primary duration')
    if exit_us > 0:
        seconds = max(reported, exit_us/1e6-start)
        basis = 'service_exit_timestamp_and_reported_enclosing_duration'
    else:
        seconds = min(request['gpu_reservation_seconds'], max(reported, now_monotonic-start))
        basis = 'conservative_observation_upper_bound_capped_by_enforced_service_limit'
    if seconds < 0 or reported > request['gpu_reservation_seconds'] or seconds > request['gpu_reservation_seconds']:
        raise ValueError('primary duration exceeds reservation or clock identity')
    return dict(resource='gpu', wall_seconds=seconds, exit_code=int(service.get('ExecMainStatus', 1)),
        status=terminal.get('status') if terminal else 'missing_terminal_result',
        accounting_basis=basis, original_denominator=96,
        nested_charges='included; never add primary slot durations', release_ready=False)


def settle_allocation(request, output, cpu_seconds):
    """Settle only this observer's receipts; preserve unrelated ledger entries."""
    import fcntl
    path = Path(request['allocation'])
    repo = Path(request['repo'])
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = read(path)
        receipts = [(output/'primary-receipt.json', 'gpu')]
        receipts += [(p, 'gpu') for p in sorted((output/'recoveries').glob('*/receipt.json'))]
        # If a local I/O failure interrupts receipt creation after issuing a
        # retry, preserve a conservative cost rather than losing attempted work.
        for attempt in read(output/'state.json')['recoveries']:
            path_for_attempt=Path(attempt['output'])/'receipt.json'
            if path_for_attempt.exists():
                continue
            conservative=output/('unsettled-'+attempt['slot_id']+'.json')
            write(conservative,dict(resource='gpu',exit_code=1,
                wall_seconds=attempt['process_cap_seconds']+request['cleanup_seconds'],
                status='missing_retry_receipt_conservative_full_cap_charge'))
            receipts.append((conservative,'gpu'))
        gpu_total=sum(read(path)['wall_seconds'] for path,resource in receipts if resource=='gpu' and path.exists())
        if gpu_total > request['gpu_reservation_seconds']:
            raise ValueError('primary plus secondary recovery exceeds original reservation')
        for receipt_path, resource in receipts:
            if not receipt_path.exists():
                continue
            relative = str(receipt_path.relative_to(repo))
            if any(row['receipt'] == relative for row in ledger['charges']):
                continue
            receipt = read(receipt_path)
            ledger['charges'].append(dict(resource=resource, seconds=receipt['wall_seconds'],
                receipt=relative, exit_code=receipt['exit_code'], nested_charges='included'))
        cpu_path = output/'receipt.json'
        relative = str(cpu_path.relative_to(repo))
        previous = next((row for row in ledger['charges'] if row['receipt'] == relative), None)
        if previous is None:
            ledger['charges'].append(dict(resource='cpu',seconds=cpu_seconds,receipt=relative,
                exit_code=read(cpu_path)['exit_code'],nested_charges='active monitoring only; sleep and GPU repair intervals excluded'))
        if (output/'primary-receipt.json').exists():
            ledger['active_reservations'] = [r for r in ledger['active_reservations']
                if r['artifact'] != request['primary_reservation_artifact']]
        ledger['active_reservations'] = [r for r in ledger['active_reservations']
            if r['artifact'] != request['monitor_reservation_artifact']]
        for resource in ('cpu', 'gpu'):
            charged = math.fsum(row['seconds'] for row in ledger['charges'] if row['resource'] == resource)
            limit = ledger['cpu_worker_seconds_reserved' if resource == 'cpu' else 'gpu_seconds_reserved']
            ledger[resource+'_seconds_charged'] = charged
            ledger[resource+'_release_remaining_seconds'] = limit-charged
            ledger[resource+'_release_uncommitted_seconds'] = limit-charged-sum(
                r['seconds'] for r in ledger['active_reservations'] if r['resource'] == resource)
            if ledger[resource+'_release_uncommitted_seconds'] < -1e-6:
                raise ValueError('settlement exceeds authorized allocation')
        ledger['last_reconciled_utc'] = datetime.now(timezone.utc).isoformat()
        write(path, ledger)


def publish(output, snapshot, state):
    record = {**snapshot, 'updated_utc': datetime.now(timezone.utc).isoformat(),
              'active_host_seconds': state['active_host_seconds'],
              'secondary_recoveries': state['recoveries'],
              'notification_errors': state['notification_errors'],
              'chat_push_available': False}
    write(output/'status.json', record)
    active = snapshot.get('active_slot') or {}
    text = ('# HMC confirmation monitor\n\n'
        f"Updated: {record['updated_utc']}\n\nStatus: **{record['status']}**. "
        f"Complete deliveries: {record.get('complete_delivery_count', 0)}/96; "
        f"recorded outcomes: {record.get('outcomes_recorded', 0)}/96.\n\n"
        f"Active slot: {active.get('slot_id', 'none')}. "
        f"Progress age: {active.get('progress_age_seconds', 'N/A')} seconds.\n\n"
        f"Resource deferrals: {', '.join(record.get('resource_deferred_slots', [])) or 'none'}.\n\n"
        f"Other failures: {json.dumps(record.get('other_failures', []))}.\n\n"
        f"Free disk bytes: {record.get('free_disk_bytes', 'not checked')}.\n\n"
        f"Error: {record.get('error', 'none')}.\n\n"
        'Original primary outcomes remain authoritative. Recovery attempts are '
        'secondary evidence and never replace primary failures. Release remains '
        'pending the independent terminal audit.\n\n'
        'Desktop alerts are attempted for errors and completion; notification '
        'failures are recorded in status.json. This observer cannot send a new '
        'message to the Codex chat.\n')
    temporary = output/'status.md.tmp'
    temporary.write_text(text)
    temporary.replace(output/'status.md')


def notify(output, state, key, title, message, *, critical=False):
    if key in state['notified']:
        return
    state['notified'].append(key)
    event = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),key=key,title=title,message=message)
    try:
        completed = subprocess.run(['notify-send', '--app-name=BayesFilter',
            '--urgency='+('critical' if critical else 'normal'), title, message],
            capture_output=True, text=True, timeout=5)
        if completed.returncode:
            raise RuntimeError(completed.stderr[-500:])
        event['desktop_delivery_requested'] = True
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        event['notification_error'] = str(error)
        state['notification_errors'].append(event)
    with (output/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(event)+'\n')


def recover(request, output, state, slot_id, remaining):
    root = Path(request['run_root'])
    design = read(request['design'])
    cap = design['search_cap_seconds']+design['closeout_cap_seconds']+design['readiness']['wait_cap_seconds']
    if remaining < cap+request['cleanup_seconds']:
        raise ValueError('unfunded resource retry')
    directory = output/'recoveries'/slot_id
    directory.mkdir(parents=True, exist_ok=False)
    original = root/slot_id/'configuration.json'
    configuration = directory/'configuration.json'
    configuration.write_bytes(original.read_bytes())
    command = [request['python'], str(root/'runner.py'), '--source', request['source'],
        '--output', str(directory), '--config', str(configuration), '--gpu', design['gpu_uuid'],
        '--design', request['design']]
    state['recoveries'].append(dict(slot_id=slot_id,status='issued',original_primary_result='resource_deferred',
        configuration_sha256=sha(configuration),output=str(directory),process_cap_seconds=cap,
        primary_outcome_unchanged=True))
    write(output/'state.json', state)
    write(directory/'launch.json', dict(command=command,cap_seconds=cap,secondary_evidence_only=True))
    environment = os.environ.copy()
    environment.update(TF_FORCE_GPU_ALLOW_GROWTH='true',CUDA_VISIBLE_DEVICES='-1',PYTHONPATH='',PYTHONUNBUFFERED='1')
    started = time.monotonic()
    code = 1
    error = None
    process = None
    try:
        with (directory/'run.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                env=environment, start_new_session=True)
            try:
                code = process.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                code = 124
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=request['cleanup_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    seconds = time.monotonic()-started
    result = dict(slot_id=slot_id,exit_code=code,error=error,wall_seconds=seconds,
        resource='gpu',command=command,original_primary_result='resource_deferred',
        secondary_evidence_only=True,primary_outcome_unchanged=True,release_ready=False,
        status='worker_completed_pending_audit' if code == 0 else 'recovery_unsuccessful')
    write(directory/'receipt.json',result)
    state['recoveries'][-1].update(result)
    write(output/'state.json',state)
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    request=read(args.request)
    output=Path(request['output'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'observer.py').write_bytes(Path(__file__).read_bytes())
    write(output/'request.json',request)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_FORCE_GPU_ALLOW_GROWTH='true')
    os.sched_setaffinity(0,{20,21,22,23})
    state=dict(active_host_seconds=0.,recoveries=[],notified=[],notification_errors=[])
    def terminated(signum, frame):
        raise RuntimeError('monitor received termination signal '+str(signum))
    signal.signal(signal.SIGTERM,terminated)
    manifest=dict(command=sys.argv,environment=sys.executable,started_utc=datetime.now(timezone.utc).isoformat(),
        request_sha256=sha(args.request),observer_sha256=sha(__file__),resource='cpu_observer',
        gpu_intentionally_hidden_in_observer=True,framework_imports=False,
        plan=request['plan'],git_commit=request['git_commit'])
    write(output/'manifest.json',manifest)
    code=0
    snapshot={}
    cycle=None
    try:
        while True:
            cycle=time.monotonic()
            service=service_status(request['service_unit'])
            snapshot=inspect_run(request,service)
            try:
                snapshot['gpu_snapshot']=gpu_snapshot(read(request['design'])['gpu_uuid'])
            except (OSError,RuntimeError,subprocess.SubprocessError) as error:
                snapshot['gpu_probe_error']=str(error)
                notify(output,state,'gpu_probe_error','HMC resource probe unavailable',
                    'GPU resource inspection failed; existing worker validity checks and caps remain in force. '+str(error),critical=True)
            notify(output,state,'started','HMC monitoring enabled',
                'Progress, artifacts, disk space and service exits are checked every minute. See '+str(output/'status.md'))
            for sid in snapshot['resource_deferred_slots']:
                notify(output,state,'deferred:'+sid,'HMC resource deferral',
                    sid+' did not start sampling. Original failure retained; one secondary resource retry may run after the primary campaign.')
            for row in snapshot['other_failures']:
                notify(output,state,'failure:'+row['slot_id'],'HMC slot error',json.dumps(row),critical=True)
            active=snapshot['active_slot'] or {}
            if active.get('warning'):
                notify(output,state,'stale:'+active['slot_id'],'HMC progress warning',
                    active['slot_id']+' has no recent checkpoint/stage update. Existing timeout remains in force.',critical=True)
            if snapshot['disk_stop_required']:
                raise ValueError('free disk below the declared infrastructure stop floor')
            if snapshot['storage_projection_warning']:
                notify(output,state,'storage_projection','HMC storage forecast warning',
                    'Free disk is below the remaining development-footprint projection plus 20%. This is a forecast warning; the separate infrastructure floor controls stopping.',critical=True)
            if not service_active(service):
                terminal_path=Path(request['run_root'])/'result.json'
                terminal=read(terminal_path) if terminal_path.exists() else None
                receipt=primary_charge(request,service,terminal,now_monotonic=time.monotonic())
                write(output/'primary-receipt.json',receipt)
                remaining=request['gpu_reservation_seconds']-receipt['wall_seconds']
                notify(output,state,'primary_finished','HMC primary campaign ended',
                    f"{snapshot['complete_delivery_count']}/96 complete deliveries; {snapshot['status']}. Release audit is pending.",
                    critical=snapshot['status']=='error_requires_diagnosis')
                for sid in snapshot['resource_deferred_slots']:
                    # Revalidate immutable identities before each retry, and
                    # never race a still-live or restarted primary service.
                    current_service=service_status(request['service_unit'])
                    inspect_run(request,current_service)
                    if not recovery_eligible(request,sid,primary_active=service_active(current_service),
                            tried={row['slot_id'] for row in state['recoveries']},remaining_seconds=remaining):
                        continue
                    if service_active(current_service):
                        raise RuntimeError('primary service became active; no recovery may run concurrently')
                    # Charge active host work before the separately charged GPU retry.
                    state['active_host_seconds']+=time.monotonic()-cycle
                    cycle=None
                    used=recover(request,output,state,sid,remaining)
                    remaining-=used
                    cycle=time.monotonic()
                    notify(output,state,'recovery:'+sid,'HMC secondary resource retry ended',
                        sid+': '+state['recoveries'][-1]['status']+'. Original confirmation outcome is unchanged.')
                    if state['recoveries'][-1]['exit_code'] not in (0,2,3,124):
                        break
                snapshot['status']='finished_pending_audit' if snapshot['terminal_status']=='execution_complete' else 'error_requires_diagnosis'
                break
            publish(output,snapshot,state)
            write(output/'state.json',state)
            state['active_host_seconds']+=time.monotonic()-cycle
            cycle=None
            if state['active_host_seconds'] >= request['active_host_cap_seconds']:
                raise RuntimeError('monitor active-host budget exhausted; primary retains its own caps')
            time.sleep(request['poll_seconds'])
    except Exception as error:
        code=1
        snapshot.update(status='monitor_error',error=type(error).__name__+': '+str(error))
        if isinstance(error,ValueError):
            try:
                stop=subprocess.run(['systemctl','--user','stop',request['service_unit']],
                    capture_output=True,text=True,timeout=request['cleanup_seconds']+10)
                snapshot['owned_service_stop_returncode']=stop.returncode
            except (OSError,subprocess.SubprocessError) as stop_error:
                snapshot['owned_service_stop_error']=str(stop_error)
        notify(output,state,'monitor_error','HMC monitor error',snapshot['error'],critical=True)
    finally:
        if cycle is not None:
            state['active_host_seconds']+=time.monotonic()-cycle
        publish(output,snapshot,state)
        write(output/'state.json',state)
        write(output/'receipt.json',dict(manifest,exit_code=code,status=snapshot['status'],
            wall_seconds=state['active_host_seconds'],accounting='active observation/notification time; idle sleeps and GPU recovery excluded'))
        settle_allocation(request,output,state['active_host_seconds'])
    return code


if __name__=='__main__':
    raise SystemExit(main())
