"""Native conditional sequential endpoint with a dynamic complete center.

Construction preserves each block's own signature. The full center is an input,
including through every captured nested TensorFlow function; no numerical center
is stored in a Python cache or a mutable resource.
"""

import tensorflow as tf

from bayesfilter.inference.program_cache_scope import ProgramCacheScope
from bayesfilter.inference.sequential_controller_tf import SequentialController
from bayesfilter.inference.sequential_map_covariance import (
    dimension_scaled_search_count,
)

D = tf.float64


class ConditionalSequentialProgram:
    """Own a block's traced target closures and complete numerical controller."""

    def __init__(self, scalar, batched, dimension, block, *, progress=False,
                 jit_compile=True):
        self.dependency_scope = ProgramCacheScope()
        self.controller = None
        cfg, start, stop = block.sequential_config, block.start, block.stop
        width = stop - start
        search_count = dimension_scaled_search_count(width) if cfg.dimension_scaled_search else cfg.search_sample_count

        @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D)],
                     jit_compile=jit_compile, autograph=False)
        def execute(center, scale):
            from bayesfilter.inference.block_coordinate_center import (
                _block_target_program,
            )

            def conditional(row):
                values, scores, _finite = _block_target_program(scalar, dimension, start, stop, None)(center, row)
                return values, scores

            batch = None
            if batched is not None:
                def batch(rows):
                    values, scores, _finite = _block_target_program(batched, dimension, start, stop,
                        int(rows.shape[0]))(center, rows)
                    return values, scores

            # This executes once during graph construction. Numerical execution
            # calls only compiled graph functions with center as a captured operand.
            controller = SequentialController(conditional, batch, None, 1, width, cfg,
                search_count, progress=progress, jit_compile=jit_compile)
            self.controller = controller
            return controller.compiled(center[None, start:stop], scale[start:stop])

        self.compiled = execute
        with self.dependency_scope.activate():
            execute.get_concrete_function()
