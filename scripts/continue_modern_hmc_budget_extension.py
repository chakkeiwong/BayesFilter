#!/usr/bin/env python3
"""Budgeted continuation of a frozen diagnostic; no statistical policy changes.

This stdlib coordinator waits for the original outer supervisor, settles its
inclusive cost once, and delegates one resume to the original source snapshot.
It neither imports TensorFlow nor edits the snapshot or old evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

POLL_SECONDS = 30.0  # Operational status cadence, not a numerical timeout.
CLOSEOUT_SECONDS = 120.0  # Inherited closeout allowance; includes process cleanup.
MASTER_BEGIN = '<!-- BEGIN MODERN-HMC-72H-20261027 -->'
MASTER_END = '<!-- END MODERN-HMC-72H-20261027 -->'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'seconds must be nonnegative and finite')
    return value


def utcnow():
    return datetime.now(timezone.utc)


def timestamp(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, 'timezone-aware timestamp required')
    return result


def settlement(allocation, parent=None, extension_seconds=0.0):
    """Pure accounting: child costs are already included in their parent."""
    inherited = seconds(allocation['inherited_unsettled_seconds'])
    credit = seconds(allocation['credit_seconds'])
    diagnostic = math.fsum(seconds(row['elapsed_seconds'])
                           for row in allocation['diagnostic_charges'])
    settled_parent = allocation.get('prior_parent_already_settled', False)
    require(type(settled_parent) is bool, 'invalid prior settlement flag')
    spent = seconds(parent['elapsed_seconds']) if parent is not None and not settled_parent else 0.0
    if parent is not None:
        require(parent.get('active_run') is False, 'cannot settle a live parent')
        require(parent['max_wall_seconds'] == allocation['prior_reserved_seconds'],
                'prior reservation changed')
    extension = seconds(extension_seconds)
    balance = inherited + credit - diagnostic - spent - extension
    reserved = inherited if parent is None and not settled_parent else 0.0
    return dict(credit_seconds=credit, inherited_unsettled_seconds=inherited,
        diagnostic_charged_seconds=diagnostic, prior_outer_charged_seconds=spent,
        extension_inclusive_charged_seconds=extension,
        remaining_seconds=max(0.0, balance), shortfall_seconds=max(0.0, -balance),
        prior_reservation_unsettled_seconds=reserved,
        unreserved_seconds=max(0.0, balance-reserved),
        separate_sir_seconds=allocation['separate_sir_seconds'],
        child_times_added=False, overlapping_passive_watch_charged=False)


def child_terminal_action(receipt, evidence):
    """Preserve a worker failure even when it precedes checkpoint creation."""
    require(receipt.get('active_run') is False, 'resume child still active')
    result_path = evidence/'result.json'
    result = read(result_path) if result_path.exists() else None
    if result and result.get('status') in {'error', 'shared_invalidity', 'paused_infrastructure'}:
        detail = result.get('error') or result.get('execution') or result['status']
        raise RuntimeError('resume worker ' + result['status'] + ': ' + str(detail))
    require(receipt.get('status') in {'completed', 'resource_stopped', 'wall_budget_exhausted'},
            'resume supervisor failure: ' + str(receipt.get('error') or receipt.get('status')))
    checkpoint = read(evidence/'tuning/tuning_checkpoint.json')
    return terminal_action(receipt, receipt, result, checkpoint)


def terminal_action(outer, child, result, checkpoint):
    """Never infer termination from an intermediate partial-budget checkpoint."""
    if outer.get('active_run') is True:
        return 'wait'
    require(outer.get('active_run') is False, 'outer lacks explicit terminal state')
    require(child.get('active_run') is False, 'qualification has not terminated')
    require(outer.get('status') == child.get('status'), 'outer/child terminal mismatch')
    status = child.get('status')
    require(status in {'completed', 'resource_stopped', 'wall_budget_exhausted'},
            'failure requires diagnosis, not automatic resume: ' + str(status))
    state = checkpoint['result']['completion_status']
    if result is not None:
        require(result.get('status') == status, 'result and supervisor disagree')
        require(result.get('execution', {}).get('completion_status') == state,
                'result and checkpoint disagree')
    if status == 'completed':
        require(result is not None and state == 'complete', 'incomplete completion evidence')
        require(child.get('exit_code') == 0, 'completed child did not exit successfully')
        return 'audit'
    require(state in {'partial_budget', 'complete'}, 'checkpoint is not resumable')
    require(status == 'wall_budget_exhausted' or result is not None,
            'resource stop lacks a result')
    return 'resume'


def validate_sources(campaign):
    source = read(campaign/'isolated-source.json')
    roots = (Path(source['source_root']).resolve(), Path(source['consumer_root']).resolve())
    require(bool(source['source_sha256']), 'empty source inventory')
    for name, expected in source['source_sha256'].items():
        path = Path(name).resolve()
        require(any(path.is_relative_to(root) for root in roots), 'source outside snapshot')
        require(digest(path) == expected, 'frozen source changed: ' + name)
    return source


def resume_command(allocation, allowance):
    require(seconds(allowance) > 1020, 'allowance must include frozen export/closeout reserves')
    campaign = Path(allocation['prior_campaign'])
    source = validate_sources(campaign)
    root = Path(source['source_root'])
    output = Path(allocation['output'])/'qualification-resume-01'
    require(not output.exists(), 'recovery output already exists')
    command = [sys.executable, str(root/'scripts/run_modern_hmc_macrofinance_frozen_diagnostic.py'),
        '--campaign', str(campaign), '--source', str(campaign/'qualification-01/evidence'),
        '--output', str(output), '--wall-seconds', str(allowance)]
    if allocation.get('recovery_entrypoint'):
        entrypoint = Path(allocation['recovery_entrypoint'])
        require(digest(entrypoint) == allocation['recovery_entrypoint_sha256'],
                'recovery host entrypoint changed')
        command[1] = str(entrypoint)
        command += ['--plan', allocation['plan']]
    env = dict(os.environ, PYTHONPATH=str(root),
        BAYESFILTER_DIAGNOSTIC_PLAN=allocation['plan'],
        BAYESFILTER_DIAGNOSTIC_CONSUMER_ROOT=source['consumer_root'],
        BAYESFILTER_DIAGNOSTIC_GPU_EVIDENCE=str(campaign/'preflight/evidence'),
        BAYESFILTER_OP_LIB_DIR=str(root/'bayesfilter/ops'),
        TF_FORCE_GPU_ALLOW_GROWTH='true', PYTHONDONTWRITEBYTECODE='1',
        TF_CPP_MIN_LOG_LEVEL='2', TF_NUM_INTRAOP_THREADS='2',
        TF_NUM_INTEROP_THREADS='2', OMP_NUM_THREADS='2')
    # The frozen supervisor reads the recorded UUID and supplies visibility.
    env.pop('CUDA_VISIBLE_DEVICES', None)
    return command, env, root, output


def terminal_audit(campaign, evidence):
    """Cross-check saved inventories; the worker owns numerical validation."""
    source = validate_sources(campaign)
    report, manifest = read(evidence/'result.json'), read(evidence/'manifest.json')
    checkpoint = read(evidence/'tuning/tuning_checkpoint.json')
    saved = checkpoint['result']
    require(report['status'] == 'completed' and saved['completion_status'] == 'complete',
            'terminal qualification is incomplete')
    require(report['sources_unchanged'] is True, 'worker found source drift')
    for name, sha in manifest['source_sha256'].items():
        require(digest(name) == sha, 'numerical dependency changed: ' + name)
    for name, path in report['loaded_local_modules'].items():
        require(str(path) in source['source_sha256'], 'import escaped frozen source: ' + name)
    runtime = read(evidence/'tuning/execution_spec.json')['execution']
    require(runtime['config']['use_xla'] is True and runtime['runtime_policy']['device_type'] == 'GPU',
            'GPU/XLA runtime changed')
    require(runtime['runtime_policy']['cuda_visible_devices'] == manifest['physical_gpu']['uuid'],
            'GPU UUID changed')
    require(manifest['memory_policy']['all_physical_devices_memory_growth'] is True,
            'memory growth unverified')
    require(all(report[key] is False for key in ('posterior_qualification',
            'external_consumer_admitted', 'default_promoted')) and report['no_ranking'] is True,
            'diagnostic acquired unsupported authority')
    candidates = saved['candidates']
    cohort = read(campaign/'frozen-cohort.json')
    expected = [(length, epsilon) for length, eps in cohort['epsilon_by_l'] for epsilon in eps]
    require(len(candidates) == len(expected)
            and {(c['leapfrog_steps'], c['epsilon']) for c in candidates} == set(expected),
            'candidate inventory differs from frozen design')
    ids = [c['candidate_id'] for c in candidates]
    verified = saved['verified_candidate_ids']
    require(len(ids) == len(set(ids)) and len(verified) == len(set(verified))
            and set(verified) <= set(ids), 'duplicate or unknown candidate')
    require(saved['candidate_states'] == report['execution']['candidate_states'],
            'candidate states disagree')
    members = read(evidence/'members.json')
    reload = report['reload']
    require(set(verified) == set(members) == set(report['execution']['verified'])
            == set(reload['verified']), 'verified/exported/reloaded membership differs')
    require(report['execution']['all_verified_members_exported'] is True
            and reload['all_verified_members_reloaded'] is True, 'member lifecycle incomplete')
    rows = reload['rows']
    require(len(rows) == len(verified) and {r['candidate_id'] for r in rows} == set(verified),
            'reload omitted or duplicated members')
    for row in rows:
        require(row['endpoint_preserved'] is True and row['finite_continuation'] is True
                and Path(row['archive']).is_file(), 'invalid reload continuation')
    require(all(Path(p).is_file() for p in members.values()), 'missing exported member')
    audit = read(evidence/'native-audit.json')
    require(audit['rhat_role'] == 'reporting_only', 'R-hat acquired tuning authority')
    require(set(audit['verified_candidate_ids']) == set(verified)
            and audit['candidate_states'] == saved['candidate_states'],
            'native audit disagrees with candidate inventory')
    require(audit['unobserved_attempted_chunk_count'] == 0,
            'completed qualification has unobserved native calls')
    # Preserve the full worker audit without inventing a second statistical test.
    return dict(candidate_count=len(candidates), verified_count=len(verified),
        verified=verified, candidate_states=saved['candidate_states'],
        source_file_count=len(source['source_sha256']),
        worker_native_audit=str(evidence/'native-audit.json'),
        result=str(evidence/'result.json'), rhat_role=audit['rhat_role'],
        posterior_qualification=False, default_promoted=False, ranking_supported=False)


def update_master(path, result, balance):
    """Replace only this allocation's status paragraph; preserve other work."""
    path = Path(path)
    body = path.read_text()
    require(body.count(MASTER_BEGIN) == body.count(MASTER_END) == 1,
            'master status markers missing or duplicated')
    before, remainder = body.split(MASTER_BEGIN)
    _, after = remainder.split(MASTER_END)
    plan_link = os.path.relpath(result.get('plan', path.parent/'bayesfilter-modern-hmc-budget-extension-2026-10-27.md'),path.parent)
    evidence_link = (os.path.relpath(Path(result['settlement']).parent,path.parent)
                     if result.get('settlement') else 'artifacts/modern-hmc-implementation-2026-10-07/N5/budget-extension-20261027-01/')
    text = (f'{MASTER_BEGIN}\n\n## Additional 72-hour allocation — 27 October 2026\n\n'
        f'Continuation status: **{result["status"]}**. '
        f'Next action: {result.get("next_action", "inspect campaign result")}. '
        f'The remaining combined allocation is {balance["remaining_seconds"]:.3f} seconds; '
        'separate SIR funding is unchanged. See the '
        f'[active continuation plan]({plan_link}) and '
        f'[live status and terminal result]({evidence_link}). '
        'This diagnostic does not establish posterior, ranking, default or release readiness.\n\n'
        f'{MASTER_END}')
    path.write_text(before+text+after)


def stop_owned_child(process, output, command):
    """Allow the existing supervisor to stop its independently-sessioned worker."""
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        receipt = read(output/'manifest.json') if (output/'manifest.json').exists() else {}
        pid = receipt.get('worker_pid')
        if type(pid) is int:
            path = Path(f'/proc/{pid}/cmdline')
            if path.exists():
                argv = path.read_bytes().decode().split('\0')
                require(command[1] in argv and str(output) in argv and '--worker' in argv,
                        'worker identity changed; manual cleanup required')
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        process.kill()
        process.wait(timeout=10)


def run_child(command, env, root, output, cap, mark, poll_seconds=POLL_SECONDS):
    began = time.monotonic()
    process = None
    try:
        with (output.parent/'resume-supervisor.log').open('x') as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                elapsed = time.monotonic()-began
                mark(dict(pid=process.pid, elapsed_seconds=elapsed,
                          supervisor_manifest=str(output/'manifest.json')))
                if elapsed >= cap:
                    raise TimeoutError('continuation supervisor exceeded its inclusive allowance')
                time.sleep(min(poll_seconds, max(0.01, cap-elapsed)))
        return process.returncode
    finally:
        stop_owned_child(process, output, command)


def result_note(result, balance):
    audit = result.get('audit', {})
    status = result['status']
    return f"""# Modern HMC budget-extension result

Status: {status}. Verified members: {audit.get('verified_count', 'not audited')}.
Remaining combined allowance: {balance['remaining_seconds']:.3f} seconds.
Prior outer time and the continuation are each charged once; child costs are included.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| {status} | {'Qualification/export/reload audited' if status == 'completed' else 'Incomplete; inspect structured result'} | {'Saved evidence checks passed' if status == 'completed' else 'See error or resource classification'} | Posterior and public admission remain open | {result.get('next_action', 'Inspect structured result')} | Posterior convergence, method ranking, default or release readiness |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Candidate outcomes remain in the numerical result; engineering errors are not relabeled as candidate rejection. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Acceptance, timing and progress comparisons. |
| Default-readiness | Not established. |
| Next evidence needed | Public admission, posterior/reference and precision evidence under the master. |

Post-run red-team: a passing orchestration audit checks preserved evidence, not
independent target correctness. Candidate rejection does not reject the research
direction. The weakest remaining inference is downstream posterior validity.
"""


def run(allocation_path):
    allocation = read(allocation_path)
    out, campaign = Path(allocation['output']), Path(allocation['prior_campaign'])
    require(not (out/'monitor-manifest.json').exists(), 'coordinator already has a run record')
    require(digest(out/'prior-settlement.json') == allocation['prior_ledger_sha256'],
            'saved prior allocation changed')
    require(allocation['numerical_attempt_limit'] == 1, 'only one automatic resume is planned')
    manifest = dict(status='waiting_for_parent', active_run=True, plan=allocation['plan'],
        command=[sys.executable, *sys.argv], started_utc=utcnow().isoformat(),
        coordinator_sha256=digest(__file__), numerical_source=str(campaign/'isolated-source.json'),
        git_commit=allocation.get('coordinator_git_commit'),
        allocation=str(allocation_path), result=str(out/'result.json'),
        python=sys.executable, parent_campaign=str(campaign))
    parent, extension_start = None, None
    result = dict(status='error', plan=allocation['plan'], posterior_qualification=False,
                  default_promoted=False, ranking_supported=False)

    def mark(details=None):
        manifest.update(heartbeat_utc=utcnow().isoformat())
        if details is not None:
            manifest['continuation'] = details
        if manifest['active_run']:
            elapsed = time.monotonic()-extension_start if extension_start is not None else 0.0
            balance = settlement(allocation, parent, elapsed)
            balance.update(reserved_for_continuation_seconds=balance['unreserved_seconds'],
                           unreserved_seconds=0.0, active_coordinator=True)
            write(out/'settlement.json', balance)
        write(out/'monitor-manifest.json', manifest)

    def interrupted(signum, frame):
        raise InterruptedError('coordinator received signal ' + str(signum))

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    mark()
    try:
        while True:
            prior = read(campaign/'manifest.json')
            manifest['parent_status'] = prior['status']
            manifest['parent_heartbeat_utc'] = prior.get('heartbeat_utc')
            mark()
            if prior.get('active_run') is not True:
                require(prior.get('active_run') is False, 'parent active state unavailable')
                parent = prior
                origin = (manifest['started_utc'] if allocation.get('prior_parent_already_settled')
                          else parent['finished_utc'])
                gap = (utcnow()-timestamp(origin)).total_seconds()
                require(gap >= -1, 'parent finish timestamp is in the future')
                extension_start = time.monotonic()-max(0.0, gap)
                break
            elapsed = (utcnow()-timestamp(prior['started_utc'])).total_seconds()
            if elapsed > seconds(prior['max_wall_seconds'])+CLOSEOUT_SECONDS:
                raise TimeoutError('parent failed to close by its deadline; no overlapping resume launched')
            time.sleep(POLL_SECONDS)
        evidence = campaign/'qualification-01/evidence'
        saved = read(evidence/'tuning/tuning_checkpoint.json')
        report = read(evidence/'result.json') if (evidence/'result.json').exists() else None
        action = terminal_action(parent, read(evidence.parent/'manifest.json'), report, saved)
        write(out/'parent-terminal.json', parent)
        write(out/'settlement.json', settlement(allocation, parent,
                                              time.monotonic()-extension_start))
        if action == 'resume':
            available = settlement(allocation, parent, time.monotonic()-extension_start)['remaining_seconds']
            if available <= 1020+CLOSEOUT_SECONDS:
                result.update(status='resource_stopped',
                              next_action='review preserved checkpoint and remaining allocation')
                return result
            command, env, root, child_out = resume_command(allocation, available-CLOSEOUT_SECONDS)
            manifest.update(status='resuming', resume_command=command, environment={key:env[key]
                for key in ('PYTHONPATH', 'BAYESFILTER_DIAGNOSTIC_CONSUMER_ROOT',
                            'BAYESFILTER_DIAGNOSTIC_GPU_EVIDENCE', 'BAYESFILTER_OP_LIB_DIR',
                            'TF_FORCE_GPU_ALLOW_GROWTH', 'PYTHONDONTWRITEBYTECODE',
                            'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS')})
            mark()
            code = run_child(command, env, root, child_out,
                             available-CLOSEOUT_SECONDS, mark)
            receipt = read(child_out/'manifest.json')
            evidence = child_out/'evidence'
            next_action = child_terminal_action(receipt, evidence)
            require(code == receipt.get('exit_code') or next_action == 'resume',
                    'process exit and supervisor disagree')
            if next_action == 'resume':
                result.update(status='resource_stopped', evidence=str(evidence),
                              next_action='review preserved checkpoint and remaining allocation')
                return result
        result.update(status='completed', evidence=str(evidence),
                      audit=terminal_audit(campaign, evidence),
                      next_action='C2 exact-source public admission review')
        if not result['audit']['verified_count']:
            result['next_action'] = 'review candidate failures before any downstream admission'
    except Exception as exc:
        result.update(status='repair_required', error=repr(exc), traceback=traceback.format_exc(),
                      next_action='diagnose localized failure; no blind automatic retry')
        write(out/'repair-alert.json', result)
    finally:
        elapsed = time.monotonic()-extension_start if extension_start is not None else 0.0
        balance = settlement(allocation, parent, elapsed)
        result.update(finished_utc=utcnow().isoformat(), settlement=str(out/'settlement.json'))
        manifest.update(status=result['status'], active_run=False, finished_utc=result['finished_utc'])
        write(out/'settlement.json', balance)
        write(out/'result.json', result)
        (out/'result.md').write_text(result_note(result, balance))
        mark()
        if allocation.get('master_plan'):
            try:
                update_master(allocation['master_plan'], result, balance)
            except Exception as exc:
                write(out/'master-update-error.json', dict(error=repr(exc), result=str(out/'result.json')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.allocation.resolve())
    print(json.dumps({key:result.get(key) for key in ('status', 'evidence', 'error')}), flush=True)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
