"""Diagnostic audits of modern-HMC source provenance and attempted native work."""
from __future__ import annotations

import ast
from pathlib import Path


def source_dependencies(entries, *, repository_root):
    """Conservative local static import closure; no imports or code execution.

    Explicit entries must include factories chosen dynamically and the actual
    adapter class module. This closes ordinary and relative imports recursively,
    including imports inside functions. It is not an execution profiler.
    """
    root = Path(repository_root).resolve()
    pending = [Path(p).resolve() for p in entries]
    seen = set()

    def local_module(parts):
        path = root.joinpath(*parts)
        for candidate in (path.with_suffix('.py'), path / '__init__.py'):
            if candidate.is_file():
                pending.append(candidate.resolve())
                return

    while pending:
        path = pending.pop()
        if path in seen:
            continue
        path.relative_to(root)
        seen.add(path)
        relative = path.relative_to(root)
        package = relative.parts[:-1]
        if package and package[0] == 'bayesfilter':
            for depth in range(1, len(package)+1):
                init = root.joinpath(*package[:depth], '__init__.py')
                if init.is_file() and init not in seen:
                    pending.append(init)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.startswith('bayesfilter.'):
                        local_module(name.name.split('.'))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = (*package[:len(package)-node.level+1],
                            *(node.module.split('.') if node.module else ()))
                else:
                    base = tuple((node.module or '').split('.'))
                if base and base[0] == 'bayesfilter':
                    local_module(base)
                    for name in node.names:
                        if name.name != '*':
                            local_module((*base, name.name))
    return tuple(str(p) for p in sorted(seen))


def audit_native_evidence(result, binding):
    """Reconcile actual trace calls, independent trials and durable charges.

    Supports partial work: attempted calls lacking a returned chunk remain
    charged and explicitly unobserved. They do not acquire successful evidence.
    """
    import tensorflow as tf
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_from_payload, _trace_from_payload
    from bayesfilter.inference.hmc_transition_spec import candidate_gradient_work
    from bayesfilter.inference.hmc_acceptance_trials import trial_seed
    from bayesfilter.inference.hmc_candidate_runtime import chunk_seed

    candidates = {c.candidate_id: c for c in result.candidates}
    works = {w.work_item_id: w for w in result.work_items}
    charges = {}
    for event in result.accounting_events:
        if event.get('event') != 'numerical_chunk_charged':
            continue
        candidate = candidates[works[event['work_item_id']].candidate_id]
        if event['gradient_work'] != event['transitions'] * candidate_gradient_work(candidate):
            raise ValueError('incorrect typed work charge')
        key = (event['work_item_id'], event['trial_ordinal'], event['trial_chunk_index'])
        if key in charges:
            raise ValueError('duplicate attempted trial chunk')
        charges[key] = event

    evidence_by_work = {row['work']['work_item_id']: row for row in binding._evidence.values()}
    chunks_by_work = {key: row['chunks'] for key, row in evidence_by_work.items()}
    chunks_by_work.update(binding._partial)
    seen_keys, seeds, rows = set(), set(), []
    completed_expected = 0
    for work in result.work_items:
        candidate = candidates[work.candidate_id]
        chunks = chunks_by_work.get(work.work_item_id, [])
        if work.status == 'completed':
            if work.work_item_id not in evidence_by_work:
                raise ValueError('completed work is missing raw evidence')
            if 'execution_failure' not in evidence_by_work[work.work_item_id]:
                expected = list(range(*work.trial_range))
                if [c['trial_ordinal'] for c in chunks] != expected:
                    raise ValueError('incomplete independent trial coverage')
                completed_expected += sum(c['count'] for c in chunks) * 4 * candidate_gradient_work(candidate)
        for chunk in chunks:
            key = (work.work_item_id, chunk['trial_ordinal'], chunk['trial_chunk_index'])
            if key in seen_keys or key not in charges:
                raise ValueError('duplicate or uncharged observed chunk')
            seen_keys.add(key)
            expected_seed = chunk_seed(trial_seed(binding.work_seed(work), chunk['trial_ordinal']),
                                       chunk['trial_chunk_index'])
            seed = tuple(chunk['seed'])
            if seed != expected_seed or seed != tuple(charges[key]['seed']) or seed in seeds:
                raise ValueError('trial stream mismatch or reuse')
            seeds.add(seed)
            if chunk['count'] != binding.config.acceptance_policy.trial_num_results + binding.config.num_warmup_steps:
                raise ValueError('incomplete whole-trial chunk')
            if chunk['initial_state'] != binding._spec['initial_active_state']:
                raise ValueError('trial did not restart at the frozen bank')
            trace = _trace_from_payload(chunk['trace'])
            runtime = chunk['runtime']
            spec = candidate.transition_spec
            if runtime.get('transition_spec') != spec.payload():
                raise ValueError('native runner transition differs from candidate')
            if runtime.get('use_xla') != binding.config.use_xla:
                raise ValueError('native compilation mode mismatch')
            device = chunk['samples_device']
            if binding._runtime['device_type'] not in device:
                raise ValueError('native samples device mismatch')
            shape = (chunk['count'], 4)
            for name in ('trajectory_steps', 'value_score_calls_per_chain', 'target_status_calls_per_chain'):
                if name not in trace or tuple(trace[name].shape) != shape:
                    raise ValueError('missing native modern runner trace')
            steps, calls, status = (trace[k] for k in ('trajectory_steps', 'value_score_calls_per_chain', 'target_status_calls_per_chain'))
            expected_calls = tf.broadcast_to(tf.reduce_max(steps, axis=1, keepdims=True)+1, shape)
            expected_status = expected_calls if binding.config.target_status_trace_policy == 'per_chain_step' else tf.zeros_like(calls)
            if not bool(tf.reduce_all((steps >= 1) & (steps <= spec.max_steps) &
                                      (calls == expected_calls) & (status == expected_status))):
                raise ValueError('realized trajectory work mismatch')
            value_calls, status_calls = int(tf.reduce_sum(calls)), int(tf.reduce_sum(status))
            if (value_calls != runtime['value_score_row_evaluations'] or
                    status_calls != runtime['target_status_row_evaluations'] or
                    value_calls+status_calls > charges[key]['gradient_work']):
                raise ValueError('native call accounting mismatch')
            rows.append(dict(work_item_id=work.work_item_id, stage=work.stage,
                trial_ordinal=chunk['trial_ordinal'], seed=seed, elapsed_seconds=chunk['elapsed_seconds'],
                native_seconds=runtime['sample_chain_call_s'], samples_device=device,
                includes_first_runner_trace_or_compilation=runtime['sample_chain_invocation_count'] == 1,
                value_score_row_evaluations=value_calls, target_status_row_evaluations=status_calls,
                realized_steps_min=int(tf.reduce_min(steps)), realized_steps_max=int(tf.reduce_max(steps))))
    return dict(charged_gradient_work=sum(e['gradient_work'] for e in charges.values()),
                expected_completed_gradient_work=completed_expected,
                attempted_chunk_count=len(charges), returned_chunk_count=len(rows),
                unobserved_attempted_chunk_count=len(set(charges)-seen_keys),
                completed_work_item_ids=[w.work_item_id for w in result.work_items if w.status == 'completed'],
                verified_candidate_ids=list(result.verified_candidate_ids),
                candidate_states=dict(result.candidate_states), work_item_count=len(works),
                native_call_audit=rows,
                native_value_score_row_evaluations=sum(r['value_score_row_evaluations'] for r in rows),
                native_target_status_row_evaluations=sum(r['target_status_row_evaluations'] for r in rows),
                rhat_role='reporting_only')
