"""Bounded static TensorFlow graphs for host-driven NeuTra training calls."""
from collections import OrderedDict

import tensorflow as tf


class FixedShapeTrainingProgram:
    """Keep explicit signatures while supporting a small set of batch shapes.

    Four entries is a host-memory bound, not a training hyperparameter.
    Eviction affects compilation only; variables belong to the trainer.
    """

    def __init__(self, function, *, jit_compile):
        self.function = function
        self.jit_compile = bool(jit_compile)
        self.programs = OrderedDict()

    def __call__(self, *arguments):
        signature = tuple(tf.TensorSpec(value.shape, value.dtype) for value in arguments)
        if any(not spec.shape.is_fully_defined() or spec.dtype != tf.float64
               for spec in signature):
            raise ValueError("NeuTra training programs require static float64 inputs")
        program = self.programs.get(signature)
        if program is None:
            program = tf.function(self.function, input_signature=signature,
                                  jit_compile=self.jit_compile)
            self.programs[signature] = program
            if len(self.programs) > 4:
                self.programs.popitem(last=False)
        self.programs.move_to_end(signature)
        return program(*arguments)
