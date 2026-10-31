"""Bounded, explicit tensor signatures for flat numerical entry points."""

from __future__ import annotations

import inspect
import types
import weakref
from functools import lru_cache, wraps

import tensorflow as tf

_DISPATCHERS = weakref.WeakKeyDictionary()


def fixed_signature_metadata(value):
    """Identify a repository-created dispatcher without trusting copied attributes."""
    if not isinstance(value, types.FunctionType) or value not in _DISPATCHERS:
        return None
    function, code, metadata = _DISPATCHERS[value]
    if (value.__code__ is not code or value.python_function is not function
            or value._jit_compile is not True):
        raise ValueError("fixed-signature dispatcher was modified")
    return dict(metadata)


def fixed_signature_function(
    *, static_parameters=(), max_specializations=16, dtype_like=None,
    floating_dtype=None, tensor_dtypes=None,
):
    """Compile each tensor shape/dtype once, retaining at most 16 signatures.

    Static parameters must be explicitly named boolean configuration switches.
    Numeric Python scalars become tensors, so their values cannot retrace the
    kernel. Python array inputs use the endpoint's declared dtype, or the
    dtype of its named primary input. Integer/mask inputs have explicit dtype
    overrides. Existing tensors retain their dtype and the endpoint's usual
    type validation. No numerical input is captured in the specialization cache.
    """

    def decorate(function):
        api = inspect.signature(function)
        static_names = frozenset(static_parameters)
        input_dtypes = dict(tensor_dtypes or {})
        if not (static_names | input_dtypes.keys()) <= api.parameters.keys():
            raise ValueError("unknown static parameter")
        if dtype_like is not None and dtype_like not in api.parameters:
            raise ValueError("unknown dtype anchor")
        if dtype_like is not None and floating_dtype is not None:
            raise ValueError("choose a fixed dtype or a dtype anchor")
        if not isinstance(max_specializations, int) or max_specializations < 1:
            raise ValueError("max_specializations must be positive")
        programs = weakref.WeakValueDictionary()

        @lru_cache(maxsize=max_specializations)
        def compiled(specifications, static):
            names, specs = zip(*specifications, strict=True)

            def numerical(*values):
                return function(**dict(zip(names, values, strict=True)), **dict(static))

            result = tf.function(
                numerical, input_signature=specs, jit_compile=True, autograph=False
            )
            # This weak registry reports only live compiled specializations.
            programs[id(result)] = result
            return result

        def prepare(args, kwargs):
            bound = api.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            dtype = floating_dtype or next(
                (
                    value.dtype
                    for value in values.values()
                    if (tf.is_tensor(value) or isinstance(value, tf.TensorSpec))
                    and value.dtype.is_floating
                ),
                tf.float64,
            )
            if dtype_like is not None:
                anchor = values[dtype_like]
                dtype = (
                    anchor.dtype if isinstance(anchor, tf.TensorSpec)
                    else tf.convert_to_tensor(anchor).dtype
                )
            inputs, specifications, static = [], [], []
            for name, value in values.items():
                if name in static_names:
                    if not isinstance(value, bool):
                        raise TypeError(f"{name} must be a static bool")
                    static.append((name, value))
                    continue
                if isinstance(value, tf.TensorSpec):
                    spec = tf.TensorSpec(value.shape, value.dtype)
                else:
                    target_dtype = input_dtypes.get(
                        name, tf.bool if isinstance(value, bool) else dtype
                    )
                    value = tf.convert_to_tensor(
                        value, dtype=None if tf.is_tensor(value) else target_dtype
                    )
                    spec = tf.TensorSpec(value.shape, value.dtype)
                specifications.append((name, spec))
                inputs.append(value)
            return compiled(tuple(specifications), tuple(static)), inputs

        @wraps(function)
        def dispatch(*args, **kwargs):
            program, values = prepare(args, kwargs)
            return program(*values)

        def concrete(*args, **kwargs):
            program, _ = prepare(args, kwargs)
            return program.get_concrete_function()

        def compiler_ir(*args, **kwargs):
            program, values = prepare(args, kwargs)
            return program.experimental_get_compiler_ir(*values)

        dispatch.python_function = function
        dispatch._jit_compile = True
        dispatch.get_concrete_function = concrete
        dispatch.experimental_get_compiler_ir = compiler_ir
        dispatch.experimental_get_tracing_count = lambda: sum(
            p.experimental_get_tracing_count() for p in programs.values()
        )
        dispatch.specialization_cache_info = compiled.cache_info
        _DISPATCHERS[dispatch] = (function, dispatch.__code__, {
            "jit_compile": True,
            "autograph": False,
            "signature_policy": "bounded_tensor_shape_dtype_specialization_v1",
            "max_specializations": max_specializations,
            "static_parameters": tuple(sorted(static_names)),
            "dtype_like": dtype_like,
            "floating_dtype": None if floating_dtype is None else tf.as_dtype(floating_dtype).name,
            "tensor_dtypes": tuple(sorted((name, tf.as_dtype(dtype).name)
                                          for name, dtype in input_dtypes.items())),
        })
        return dispatch

    return decorate
