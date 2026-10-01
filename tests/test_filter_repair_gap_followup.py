"""Diagnostic-only ownership and exact external supervisor source probes."""

import ast
import gc
import hashlib
import sys
import weakref
from pathlib import Path

from bayesfilter.inference.sequential_lifecycle_tf import lifecycle_program
from bayesfilter.inference.sequential_map_covariance import (
    SequentialMapCovarianceConfig,
)
from bayesfilter.inference.sequential_refinement_tf import refinement_program
from bayesfilter.inference.sequential_terminal_tf import terminal_program
from tests.test_filter_repair_geometry_control import save
from tests.test_filter_repair_structured_memory import _inputs


def inference_caches():
    caches = {}
    seen = set()
    for name, module in tuple(sys.modules.items()):
        if name.startswith('bayesfilter.inference.'):
            for key, value in vars(module).items():
                if hasattr(value, 'cache_clear') and hasattr(value, 'cache_info') and id(value) not in seen:
                    seen.add(id(value))
                    caches[f'{name}.{key}'] = value
    return caches


def test_nested_lifecycle_cache_ownership(request):
    """Tracing only: cache eviction is distinct from XLA/native memory release."""
    for cache in inference_caches().values():
        cache.cache_clear()
    scalar, batched, _ = _inputs(3, 4)
    cfg = SequentialMapCovarianceConfig(locator_policy='center_first',
        refinement_geometry_policy='factor_correlation', structured_max_factors=1,
        max_attempts=2, search_sample_count=4, terminal_sample_count=24,
        max_exact_evaluations=256)
    refine = refinement_program(scalar, batched, 3, cfg, cfg.search_sample_count)
    terminal = terminal_program(scalar, batched, 3, cfg)
    program = lifecycle_program(refine, terminal, 3, cfg, cfg.search_sample_count)
    concrete = program.get_concrete_function()
    references = {'scalar': weakref.ref(scalar), 'batched': weakref.ref(batched),
        'lifecycle': weakref.ref(program), 'graph': weakref.ref(concrete.graph)}

    def observe():
        gc.collect()
        return {'alive': {name: ref() is not None for name, ref in references.items()},
            'populated_caches': {name: cache.cache_info()._asdict()
                for name, cache in inference_caches().items() if cache.cache_info().currsize}}

    del scalar, batched, refine, terminal, program, concrete
    stages = {'caller_released': observe()}
    lifecycle_program.cache_clear()
    stages['lifecycle_cache_cleared'] = observe()
    for cache in inference_caches().values():
        cache.cache_clear()
    stages['dependency_caches_cleared'] = observe()
    save(request, 'gap-cache-ownership.json', {'role': 'diagnostic_python_cache_ownership',
        'stages': stages, 'executed_numerical_program': False,
        'limitation': 'No attribution of native executable/allocator residency; one traced lifecycle.'})
    assert all(stages['caller_released']['alive'].values())
    assert stages['lifecycle_cache_cleared']['alive']['scalar']
    assert stages['lifecycle_cache_cleared']['alive']['batched']
    assert not any(stages['dependency_caches_cleared']['alive'].values())


def test_current_external_progress_supervisor_semantics(request):
    """Execute only the inspected pure function and literal guard constants."""
    root = Path('/home/ubuntu/workspace/MacroFinance-dz5-neutra')
    path = root / 'scripts/run_two_currency_double_zlb_dz5_hierarchical_initializer.py'
    constants_path = root / 'two_currency_double_zlb_dz5_resource_policy.py'
    source = path.read_text()
    constants_source = constants_path.read_text()
    function = next(node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_progress_supervisor')
    constants = {}
    names = {'CANARY_BASELINE_GUARD_SECONDS', 'CANARY_TOTAL_GUARD_SECONDS', 'CANARY_NO_PROGRESS_SECONDS'}
    for node in ast.parse(constants_source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                constants[name] = ast.literal_eval(node.value)
    assert constants.keys() == names
    namespace = dict(constants)
    excerpt = 'from __future__ import annotations\n' + ast.get_source_segment(source, function)
    exec(compile(excerpt, str(path), 'exec'), namespace)  # noqa: S102 - exact diagnostic source excerpt
    evaluate = namespace[function.name]
    no_progress = constants['CANARY_NO_PROGRESS_SECONDS']
    cases = {
        'over_budget_advancing': {'elapsed_seconds': constants['CANARY_TOTAL_GUARD_SECONDS'] + 1.,
            'semantic_sequence': 2, 'previous_semantic_sequence': 1, 'seconds_since_semantic_progress': 0.,
            'active_stage': 'compiled_phase', 'active_stage_elapsed_seconds': constants['CANARY_BASELINE_GUARD_SECONDS'] + 1.},
        'quiet_phase_before_stage_report': {'elapsed_seconds': no_progress + 1.,
            'semantic_sequence': 0, 'previous_semantic_sequence': 0, 'seconds_since_semantic_progress': no_progress + 1.,
            'active_stage': None, 'active_stage_elapsed_seconds': None},
        'quiet_boundary': {'elapsed_seconds': no_progress, 'semantic_sequence': 0,
            'previous_semantic_sequence': 0, 'seconds_since_semantic_progress': no_progress,
            'active_stage': None, 'active_stage_elapsed_seconds': None},
    }
    observations = {name: {'input': inputs, 'output': evaluate(**inputs)} for name, inputs in cases.items()}
    save(request, 'gap-external-supervisor.json', {'role': 'diagnostic_exact_source_excerpt',
        'source': str(path), 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'function_lines': [function.lineno, function.end_lineno], 'constants': constants,
        'constants_source': str(constants_path),
        'constants_sha256': hashlib.sha256(constants_source.encode()).hexdigest(),
        'observations': observations,
        'limitation': 'No external worker launch, termination test, source mutation or consumer qualification.'})
    assert len(observations['over_budget_advancing']['output']['resource_guard_crossings']) == 2
    assert not observations['over_budget_advancing']['output']['terminate_for_stale_progress']
    assert no_progress + 1. < constants['CANARY_TOTAL_GUARD_SECONDS']
    assert not observations['quiet_phase_before_stage_report']['output']['resource_guard_crossings']
    assert observations['quiet_phase_before_stage_report']['output']['terminate_for_stale_progress']
    assert not observations['quiet_boundary']['output']['terminate_for_stale_progress']
