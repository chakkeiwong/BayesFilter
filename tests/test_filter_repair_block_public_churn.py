"""Diagnostic bounded public-block callback/shape churn and native residency.

The numerical authority remains original3582b4ac. Python collection is tested
separately from native mappings; no native eviction or leak-freedom claim.
"""

import dataclasses
import gc
import json
import time
import weakref
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference import block_controller_tf as runtime
from bayesfilter.inference import block_coordinate_center as public
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_block_public_memory import fixture
from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_lifecycle_original import _compare_original
from tests.test_filter_repair_posterior_residency import mapping_snapshot


def test_bounded_public_block_callback_and_shape_churn(request):
    gpu = bool(tf.config.list_logical_devices('GPU'))
    root = Path(request.config.getoption('xmlpath')).parent
    vm_cap = int(Path('/proc/sys/vm/max_map_count').read_text())
    observations, cases = [], []
    runtime.clear_block_controller_cache()

    def snapshot(stage, cycle):
        mappings = mapping_snapshot()
        record = {'stage': stage, 'cycle': cycle, 'memory': memory_snapshot(gpu),
            'mappings': mappings, 'map_count': sum(v['mappings'] for v in mappings.values())}
        observations.append(record)
        # Append-only stage evidence survives a later native allocation failure.
        with (root / 'block-churn-stages.jsonl').open('a') as output:
            output.write(json.dumps(record, allow_nan=False) + '\n')
        return record

    with GPUProcessMonitor(gpu) as sharing:
        for cycle, dimension in enumerate((3, 5, 3, 5)):
            before = snapshot('before_construction', cycle)
            assert before['map_count'] < .7 * vm_cap, 'Bounded diagnostic stopped before further compilation'
            scalar, batch, cfg, args = fixture(dimension)
            blocks = (public.BlockCoordinateCenterBlock('wide', 0, dimension-1, cfg),
                public.BlockCoordinateCenterBlock('last', dimension-1, dimension, cfg))
            options = public.BlockCoordinateCenterConfig(max_physical_target_rows=600,
                stop_on_material_reversal=False)
            begin = time.perf_counter()
            owner = runtime.block_controller(scalar, batch, dimension, blocks, options)
            concrete = owner.compiled.get_concrete_function()
            references = {'owner': weakref.ref(owner), 'graph': weakref.ref(concrete.graph),
                'scope': weakref.ref(owner.dependency_scope), 'scalar': weakref.ref(scalar),
                'batch': weakref.ref(batch)}
            snapshot('built_and_traced', cycle)

            def execute(values, scalar=scalar, batch=batch, blocks=blocks, options=options):
                return public.locate_block_coordinate_center(scalar, values[0], blocks=blocks,
                    batched_value_and_score_fn=batch, scale=values[1], config=options).private_payload()

            first = execute(args)
            cold_seconds = time.perf_counter() - begin
            snapshot('cold', cycle)
            changed = (args[0] + .0005, args[1] * 1.1)
            second = execute(changed)
            snapshot('changed', cycle)
            begin = time.perf_counter()
            for index in range(50):
                assert execute(args if index % 2 == 0 else changed) == (first if index % 2 == 0 else second)
                if (index + 1) % 25 == 0:
                    snapshot(f'reuse_{index+1}', cycle)
            reuse_seconds = time.perf_counter() - begin
            traces = owner.compiled.experimental_get_tracing_count()
            assert traces == 1
            runtime.clear_block_controller_cache()
            del execute, owner, concrete, scalar, batch
            gc.collect()
            released = {name: reference() is None for name, reference in references.items()}
            snapshot('python_release', cycle)
            assert all(released.values()), released
            cases.append({'dimension': dimension, 'cycle': cycle, 'trace_count': traces,
                'result': first, 'changed_result': second, 'python_released': released,
                'cold_seconds': cold_seconds, 'reuse_50_seconds': reuse_seconds})
    checkpoint = FrozenCheckpoint('3582b4ac', 'block_churn_original')
    original = checkpoint.load('bayesfilter.inference.block_coordinate_center')
    original_seq = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    for case in cases:
        dimension = case['dimension']
        scalar, batch, cfg, args = fixture(dimension)
        original_cfg = original_seq.SequentialMapCovarianceConfig(**dataclasses.asdict(cfg))
        blocks = (original.BlockCoordinateCenterBlock('wide', 0, dimension-1, original_cfg),
            original.BlockCoordinateCenterBlock('last', dimension-1, dimension, original_cfg))
        options = original.BlockCoordinateCenterConfig(max_physical_target_rows=600,
            stop_on_material_reversal=False)
        for name, values in (('result', args), ('changed_result', (args[0]+.0005, args[1]*1.1))):
            expected = original.locate_block_coordinate_center(scalar, values[0], blocks=blocks,
                batched_value_and_score_fn=batch, scale=values[1], config=options).private_payload()
            case['original_' + name] = expected
            _compare_original(case[name], expected)
    report = {'schema': 'filter_block_public_churn.v1', 'gpu': gpu, 'cases': cases,
        'observations': observations, 'vm_max_map_count': vm_cap, 'precompile_map_fraction_stop': .7,
        'gpu_process_observation': sharing.payload(), 'original_source_sha256': checkpoint.hashes(),
        'role': 'bounded_shape_and_callback_churn_allocation_attribution',
        'nonclaims': ['Four signatures and200 reuse calls cannot establish unbounded lifetime safety.',
            'Python release does not imply native eviction; observations are not exact peaks.',
            'Timing is explanatory and may be shared; not matched cost qualification.']}
    with (root / 'block-public-churn.json').open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
