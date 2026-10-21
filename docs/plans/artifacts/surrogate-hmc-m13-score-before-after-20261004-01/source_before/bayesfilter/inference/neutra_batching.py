"""Fail-closed batch-native target binding for BayesFilter NeuTra training."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import textwrap
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import value_score_capability


NEUTRA_BATCH_METHOD = "neutra_batch_log_prob_and_grad_status"
NEUTRA_BATCHING_SCHEMA = "bayesfilter.neutra.batch_native_target_binding.v2"
REQUIRED_STATUS_FIELDS = (
    "status_code",
    "valid_pre_regularized_score",
    "floor_count_value",
    "min_innovation_eigenvalue",
)
OPTIONAL_STATUS_FIELDS = ("innovation_condition_estimate",)
NORMALIZED_STATUS_FIELDS = (
    *REQUIRED_STATUS_FIELDS,
    "min_innovation_eigenvalue_available",
    *OPTIONAL_STATUS_FIELDS,
    "innovation_condition_estimate_available",
)
_ISSUER = object()
_FORBIDDEN_TF_CALLS = {
    "map_fn",
    "vectorized_map",
    "numpy_function",
    "py_function",
}


class InvalidNeuTraBatchTarget(ValueError):
    """Raised when a target is ineligible for NeuTra optimizer updates."""


@dataclass(frozen=True)
class NeuTraBatchTargetBinding:
    """Repository-issued binding to an inspected batch-native adapter method."""

    schema: str
    target_signature: str
    adapter_signature: str
    target_scope: str
    backend_id: str
    method_name: str
    callable_module: str
    callable_qualname: str
    callable_source_sha256: str
    dependency_closure_sha256: str
    dependency_module_sources: tuple[tuple[str, str], ...]
    dependency_callable_sources: tuple[tuple[str, str, str, str], ...]
    evidence_path: str | None
    minimum_batch_size: int
    jit_compile_required: bool
    status_telemetry_required: bool
    scalar_fallback_used: bool
    sample_axis_python_loop_used: bool
    row_mapped_scalar_target_used: bool
    source_audit_scope: str
    _owner: Any = field(repr=False, compare=False)
    _function: Callable[..., Any] = field(repr=False, compare=False)
    _dependency_functions: tuple[tuple[str, Callable[..., Any]], ...] = field(
        repr=False, compare=False
    )
    _issuer: object = field(repr=False, compare=False)

    def invoke(
        self, values: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        """Invoke the same bound method inspected when this binding was issued."""

        _validate_binding_integrity(self)
        current = getattr(self._owner, self.method_name)
        return current(values)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "target_signature": self.target_signature,
            "adapter_signature": self.adapter_signature,
            "target_scope": self.target_scope,
            "backend_id": self.backend_id,
            "method_name": self.method_name,
            "callable_module": self.callable_module,
            "callable_qualname": self.callable_qualname,
            "callable_source_sha256": self.callable_source_sha256,
            "dependency_closure_sha256": self.dependency_closure_sha256,
            "dependency_module_sources": [
                {"module": module, "source_sha256": source_sha256}
                for module, source_sha256 in self.dependency_module_sources
            ],
            "dependency_callable_sources": [
                {
                    "global_name": global_name,
                    "module": module,
                    "qualname": qualname,
                    "source_sha256": source_sha256,
                }
                for global_name, module, qualname, source_sha256
                in self.dependency_callable_sources
            ],
            "evidence_path": self.evidence_path,
            "minimum_batch_size": self.minimum_batch_size,
            "jit_compile_required": self.jit_compile_required,
            "status_telemetry_required": self.status_telemetry_required,
            "scalar_fallback_used": self.scalar_fallback_used,
            "sample_axis_python_loop_used": self.sample_axis_python_loop_used,
            "row_mapped_scalar_target_used": self.row_mapped_scalar_target_used,
            "source_audit_scope": self.source_audit_scope,
        }


class BoundBatchNativeNeuTraTrainingTarget:
    """Trainer-facing target that executes one repository-issued binding."""

    def __init__(self, binding: NeuTraBatchTargetBinding) -> None:
        _validate_binding_integrity(binding)
        owner = binding._owner
        required = ("config", "parameter_dim", "parameter_names")
        missing = tuple(name for name in required if not hasattr(owner, name))
        if missing:
            raise InvalidNeuTraBatchTarget(
                f"bound NeuTra training target is missing identity fields: {missing}"
            )
        self._binding = binding
        self.config = owner.config
        self.parameter_dim = int(owner.parameter_dim)
        self.parameter_names = tuple(str(name) for name in owner.parameter_names)
        self.target_scope = binding.target_scope

    def target_signature(self) -> str:
        return self._binding.target_signature

    def adapter_signature(self) -> str:
        return self._binding.adapter_signature

    def binding_payload(self) -> Mapping[str, Any]:
        return self._binding.payload()

    def batch_value_and_score(
        self, theta: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, status = self._binding.invoke(theta)
        value_tensor = tf.convert_to_tensor(value, tf.float64)
        score_tensor = tf.convert_to_tensor(score, tf.float64)
        values = tf.convert_to_tensor(theta, tf.float64)
        if value_tensor.shape != values.shape[:-1] or score_tensor.shape != values.shape:
            raise InvalidNeuTraBatchTarget("bound target value/score shape mismatch")
        valid = _hard_valid_training_status(
            status,
            value=value_tensor,
            score=score_tensor,
        )
        invalid_value = tf.fill(
            tf.shape(value_tensor), tf.constant(float("nan"), tf.float64)
        )
        invalid_score = tf.fill(
            tf.shape(score_tensor), tf.constant(float("nan"), tf.float64)
        )
        return (
            tf.where(valid, value_tensor, invalid_value),
            tf.where(valid[..., tf.newaxis], score_tensor, invalid_score),
        )

    def batch_value_score_status(
        self, theta: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        value, score, status = self._binding.invoke(theta)
        value_tensor = tf.convert_to_tensor(value, tf.float64)
        score_tensor = tf.convert_to_tensor(score, tf.float64)
        normalized = _normalized_training_status(
            status,
            value=value_tensor,
            score=score_tensor,
        )
        return value_tensor, score_tensor, normalized


def bound_batch_native_neutra_training_target(
    binding: NeuTraBatchTargetBinding,
) -> BoundBatchNativeNeuTraTrainingTarget:
    """Return the only trainer-facing proxy for an issued batch binding."""

    return BoundBatchNativeNeuTraTrainingTarget(binding)


def bind_batch_native_neutra_target(
    adapter: Any,
    *,
    target_signature: str,
) -> NeuTraBatchTargetBinding:
    """Inspect and bind the fixed adapter method eligible for NeuTra training."""

    method = getattr(adapter, NEUTRA_BATCH_METHOD, None)
    if not callable(method):
        raise InvalidNeuTraBatchTarget(
            f"NeuTra training requires bound method {NEUTRA_BATCH_METHOD!r}"
        )
    if not inspect.ismethod(method) or method.__self__ is not adapter:
        raise InvalidNeuTraBatchTarget(
            "NeuTra batch target must be an adapter-bound instance method"
        )
    function = method.__func__
    source = _method_source(function)
    _audit_direct_method_source(source)
    (
        dependency_modules,
        dependency_callables,
        dependency_functions,
        dependency_closure,
    ) = _repository_dependency_closure(
        function,
        source,
    )
    capability = value_score_capability(adapter)
    if capability.value_score_authority != "graph_native":
        raise InvalidNeuTraBatchTarget(
            "NeuTra batch target requires graph_native value/score authority"
        )
    if not capability.xla_hmc_ready:
        raise InvalidNeuTraBatchTarget("NeuTra batch target must be XLA ready")

    signature = _bare_sha256(target_signature, "target_signature")
    adapter_signature = _adapter_signature(
        adapter,
        target_signature=signature,
        callable_source=source,
        target_scope=capability.target_scope,
    )
    binding = NeuTraBatchTargetBinding(
        schema=NEUTRA_BATCHING_SCHEMA,
        target_signature=signature,
        adapter_signature=adapter_signature,
        target_scope=str(capability.target_scope or type(adapter).__qualname__),
        backend_id=str(capability.runtime_backend),
        method_name=NEUTRA_BATCH_METHOD,
        callable_module=str(function.__module__),
        callable_qualname=str(function.__qualname__),
        callable_source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        dependency_closure_sha256=dependency_closure,
        dependency_module_sources=dependency_modules,
        dependency_callable_sources=dependency_callables,
        evidence_path=(
            None if capability.evidence_path is None else str(capability.evidence_path)
        ),
        minimum_batch_size=2,
        jit_compile_required=True,
        status_telemetry_required=True,
        scalar_fallback_used=False,
        sample_axis_python_loop_used=False,
        row_mapped_scalar_target_used=False,
        source_audit_scope=(
            "bound_adapter_method_direct_source_plus_repository_module_closure_v2"
        ),
        _owner=adapter,
        _function=function,
        _dependency_functions=dependency_functions,
        _issuer=_ISSUER,
    )
    _validate_binding_integrity(binding)
    return binding


def require_batch_native_neutra_target(
    adapter: Any,
    *,
    target_signature: str,
    batch_size: int,
) -> NeuTraBatchTargetBinding:
    """Return an eligible binding or reject before any training side effect."""

    binding = bind_batch_native_neutra_target(
        adapter,
        target_signature=target_signature,
    )
    if int(batch_size) < binding.minimum_batch_size:
        raise InvalidNeuTraBatchTarget(
            f"NeuTra training batch size must be at least {binding.minimum_batch_size}"
        )
    return binding


def batch_native_value_status_target_fn(
    binding: NeuTraBatchTargetBinding,
) -> Callable[[tf.Tensor], tuple[tf.Tensor, Mapping[str, tf.Tensor]]]:
    """Attach the reviewed batch score as the target-value custom gradient."""

    _validate_binding_integrity(binding)

    def target_value_status(
        theta: tf.Tensor,
    ) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        if values.shape.rank != 2:
            raise ValueError("batch-native NeuTra target requires rank 2 theta")
        if values.shape[0] is not None and int(values.shape[0]) < binding.minimum_batch_size:
            raise ValueError("batch-native NeuTra target received a singleton batch")
        tf.debugging.assert_greater_equal(
            tf.shape(values)[0],
            tf.constant(binding.minimum_batch_size, tf.int32),
            message="batch-native NeuTra target requires batch size greater than one",
        )

        @tf.custom_gradient
        def invoke_with_reviewed_score(x: tf.Tensor):
            value, score, status = binding.invoke(tf.stop_gradient(x))
            value_tensor = tf.stop_gradient(tf.convert_to_tensor(value, tf.float64))
            score_tensor = tf.stop_gradient(tf.convert_to_tensor(score, tf.float64))
            if value_tensor.shape.rank != 1 or score_tensor.shape != x.shape:
                raise ValueError("batch-native target value/score shape mismatch")
            if not isinstance(status, Mapping):
                raise TypeError("batch-native target status must be a mapping")
            missing = tuple(
                name for name in REQUIRED_STATUS_FIELDS if name not in status
            )
            if missing:
                raise ValueError(
                    f"batch-native target status is missing fields: {missing}"
                )
            condition_present = "innovation_condition_estimate" in status
            condition_available = tf.convert_to_tensor(
                status.get(
                    "innovation_condition_estimate_available",
                    tf.fill(
                        tf.shape(value_tensor),
                        condition_present,
                    ),
                ),
                tf.bool,
            )
            min_eigen_available = tf.convert_to_tensor(
                status.get(
                    "min_innovation_eigenvalue_available",
                    tf.ones_like(value_tensor, tf.bool),
                ),
                tf.bool,
            )
            condition_estimate = (
                tf.convert_to_tensor(
                    status["innovation_condition_estimate"], tf.float64
                )
                if condition_present
                else tf.ones_like(value_tensor, tf.float64)
            )
            outputs = (
                value_tensor,
                *(
                    tf.stop_gradient(tf.convert_to_tensor(status[name]))
                    for name in REQUIRED_STATUS_FIELDS
                ),
                tf.stop_gradient(min_eigen_available),
                tf.stop_gradient(condition_estimate),
                tf.stop_gradient(condition_available),
            )

            def grad(upstream: Any, *_status_gradients: Any) -> tf.Tensor:
                dy = tf.convert_to_tensor(upstream, dtype=score_tensor.dtype)
                return dy[..., tf.newaxis] * score_tensor

            return outputs, grad

        outputs = invoke_with_reviewed_score(values)
        return outputs[0], {
            name: outputs[index + 1]
            for index, name in enumerate(NORMALIZED_STATUS_FIELDS)
        }

    return target_value_status


def _normalized_training_status(
    status: Mapping[str, Any],
    *,
    value: tf.Tensor,
    score: tf.Tensor,
) -> Mapping[str, tf.Tensor]:
    if not isinstance(status, Mapping):
        raise InvalidNeuTraBatchTarget("bound target status must be a mapping")
    missing = tuple(name for name in REQUIRED_STATUS_FIELDS if name not in status)
    if missing:
        raise InvalidNeuTraBatchTarget(
            f"bound target status is missing fields: {missing}"
        )
    status_code = tf.convert_to_tensor(status["status_code"], tf.int32)
    valid_score = tf.convert_to_tensor(
        status["valid_pre_regularized_score"], tf.bool
    )
    floor_count = tf.convert_to_tensor(status["floor_count_value"], tf.int32)
    min_eigenvalue = tf.convert_to_tensor(
        status["min_innovation_eigenvalue"], tf.float64
    )
    expected_shape = value.shape
    for name, tensor in (
        ("status_code", status_code),
        ("valid_pre_regularized_score", valid_score),
        ("floor_count_value", floor_count),
        ("min_innovation_eigenvalue", min_eigenvalue),
    ):
        if tensor.shape != expected_shape:
            raise InvalidNeuTraBatchTarget(f"bound target {name} shape mismatch")
    condition_available = "innovation_condition_estimate" in status
    condition = (
        tf.convert_to_tensor(status["innovation_condition_estimate"], tf.float64)
        if condition_available
        else tf.ones_like(value, tf.float64)
    )
    if condition.shape != expected_shape:
        raise InvalidNeuTraBatchTarget(
            "bound target innovation_condition_estimate shape mismatch"
        )
    hard_valid = tf.logical_and(
        tf.math.is_finite(value),
        tf.logical_and(
            tf.reduce_all(tf.math.is_finite(score), axis=-1),
            tf.logical_and(
                tf.equal(status_code, 0),
                tf.logical_and(
                    valid_score,
                    tf.logical_and(
                        tf.equal(floor_count, 0),
                        tf.logical_and(
                            tf.math.is_finite(min_eigenvalue),
                            min_eigenvalue > 0.0,
                        ),
                    ),
                ),
            ),
        ),
    )
    return {
        "status_code": status_code,
        "valid_pre_regularized_score": valid_score,
        "floor_count_value": floor_count,
        "min_innovation_eigenvalue": min_eigenvalue,
        "innovation_condition_estimate": condition,
        "innovation_condition_estimate_available": tf.fill(
            tf.shape(value), tf.constant(condition_available)
        ),
        "hard_valid_for_training": hard_valid,
    }


def _hard_valid_training_status(
    status: Mapping[str, Any],
    *,
    value: tf.Tensor,
    score: tf.Tensor,
) -> tf.Tensor:
    return _normalized_training_status(status, value=value, score=score)[
        "hard_valid_for_training"
    ]


def _validate_binding_integrity(binding: NeuTraBatchTargetBinding) -> None:
    if binding._issuer is not _ISSUER:
        raise InvalidNeuTraBatchTarget("NeuTra batch binding was not repository-issued")
    if binding.schema != NEUTRA_BATCHING_SCHEMA:
        raise InvalidNeuTraBatchTarget("NeuTra batch binding schema mismatch")
    current = getattr(binding._owner, binding.method_name, None)
    if not inspect.ismethod(current) or current.__self__ is not binding._owner:
        raise InvalidNeuTraBatchTarget("NeuTra batch callable is no longer owner-bound")
    if current.__func__ is not binding._function:
        raise InvalidNeuTraBatchTarget("NeuTra batch callable changed after binding")
    current_source = _method_source(current.__func__)
    current_source_sha256 = hashlib.sha256(
        current_source.encode("utf-8")
    ).hexdigest()
    if current_source_sha256 != binding.callable_source_sha256:
        raise InvalidNeuTraBatchTarget("NeuTra batch callable source changed after binding")
    (
        dependency_modules,
        dependency_callables,
        dependency_functions,
        dependency_closure,
    ) = _repository_dependency_closure(
        current.__func__,
        current_source,
    )
    if (
        dependency_modules != binding.dependency_module_sources
        or dependency_callables != binding.dependency_callable_sources
        or dependency_functions != binding._dependency_functions
        or dependency_closure != binding.dependency_closure_sha256
    ):
        raise InvalidNeuTraBatchTarget(
            "NeuTra batch repository dependency closure changed after binding"
        )
    if binding.minimum_batch_size < 2:
        raise InvalidNeuTraBatchTarget("NeuTra batch binding permits singleton batches")
    if not binding.jit_compile_required or not binding.status_telemetry_required:
        raise InvalidNeuTraBatchTarget("NeuTra batch binding lacks XLA/status requirements")
    if (
        binding.scalar_fallback_used
        or binding.sample_axis_python_loop_used
        or binding.row_mapped_scalar_target_used
    ):
        raise InvalidNeuTraBatchTarget("NeuTra batch binding declares an ineligible fallback")


def _method_source(function: Callable[..., Any]) -> str:
    try:
        return textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as exc:
        raise InvalidNeuTraBatchTarget(
            "NeuTra batch method source must be inspectable"
        ) from exc


def _audit_direct_method_source(source: str) -> None:
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            violations.append(f"python_loop:{node.lineno}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _FORBIDDEN_TF_CALLS:
                violations.append(f"forbidden_tensorflow_map_or_callback:{node.lineno}")
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                violations.append(f"adapter_method_delegation:{node.lineno}")
    if violations:
        raise InvalidNeuTraBatchTarget(
            "NeuTra batch method is row-mapped or callback-backed: "
            + ", ".join(violations)
        )


def _repository_dependency_closure(
    function: Callable[..., Any],
    source: str,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str, str, str], ...],
    tuple[tuple[str, Callable[..., Any]], ...],
    str,
]:
    tree = ast.parse(source)
    called_names = tuple(
        sorted(
            {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
        )
    )
    dependencies = tuple(
        sorted(
            (
                name,
                function.__globals__[name],
            )
            for name in called_names
            if callable(function.__globals__.get(name))
            and not inspect.isclass(function.__globals__.get(name))
            and str(
                getattr(function.__globals__.get(name), "__module__", "")
            ).startswith("bayesfilter.")
        )
    )
    repository_modules = tuple(
        sorted(
            {
                str(getattr(dependency, "__module__", ""))
                for _name, dependency in dependencies
            }
        )
    )
    module_sources = tuple(
        (
            module_name,
            hashlib.sha256(
                _module_source(module_name).encode("utf-8")
            ).hexdigest(),
        )
        for module_name in repository_modules
    )
    callable_sources = tuple(
        (
            global_name,
            str(dependency.__module__),
            str(dependency.__qualname__),
            hashlib.sha256(
                _method_source(dependency).encode("utf-8")
            ).hexdigest(),
        )
        for global_name, dependency in dependencies
    )
    dependency_functions = tuple(dependencies)
    encoded = json.dumps(
        {
            "module_sources": module_sources,
            "callable_sources": callable_sources,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        module_sources,
        callable_sources,
        dependency_functions,
        hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )


def _module_source(module_name: str) -> str:
    module = sys.modules.get(module_name)
    if module is None:
        raise InvalidNeuTraBatchTarget(
            f"NeuTra batch dependency module is not loaded: {module_name}"
        )
    try:
        return inspect.getsource(module)
    except (OSError, TypeError) as exc:
        raise InvalidNeuTraBatchTarget(
            f"NeuTra batch dependency module source is not inspectable: {module_name}"
        ) from exc


def _adapter_signature(
    adapter: Any,
    *,
    target_signature: str,
    callable_source: str,
    target_scope: str | None,
) -> str:
    signature_method = getattr(adapter, "adapter_signature", None)
    if callable(signature_method):
        return _bare_sha256(signature_method(), "adapter_signature")
    payload = {
        "adapter_module": type(adapter).__module__,
        "adapter_qualname": type(adapter).__qualname__,
        "target_signature": target_signature,
        "target_scope": target_scope,
        "callable_source_sha256": hashlib.sha256(
            callable_source.encode("utf-8")
        ).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _bare_sha256(value: Any, label: str) -> str:
    text = str(value)
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise InvalidNeuTraBatchTarget(f"{label} must be a lowercase SHA-256 digest")
    return text
