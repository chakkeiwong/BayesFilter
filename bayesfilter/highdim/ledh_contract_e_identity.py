"""Factory-bound semantic identity for canonical Contract E LEDH candidates.

Phase 2 intentionally leaves the public production registry empty.  Later
phases may register exact BayesFilter-owned implementation symbols only after
their own evidence gates pass.
"""

from __future__ import annotations

import base64
import builtins
import dis
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import platform
import sys
import types
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


CONTRACT_E_ROUTE_IDENTITY_SCHEMA_VERSION = (
    "bayesfilter.highdim.contract_e_route_identity.v2"
)
CONTRACT_E_ROUTE_FACTORY_ID = (
    "bayesfilter.highdim.contract_e_canonical_route_factory.v1"
)
CONTRACT_E_RESET_CONTRACT_ID = "contract_e_chol_v1"
CONTRACT_E_DERIVATIVE_COMPOSITION_ID = (
    "contract_e_chol_total_direct_moments_weights_plus_streaming_transport_v1"
)
CONTRACT_E_ROW_NORMALIZATION_POLICY_ID = (
    "streaming_positive_transport_row_mass_quotient_v1"
)
CONTRACT_E_PHASE2_CANDIDATE_STATUS = (
    "factory_bound_identity_candidate_not_admitted_phase2"
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_FACTORY_CONSTRUCTION_KEY = object()
_AUTHORIZED_FACTORY_SEALS: dict[object, tuple[str, object]] = {}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(name): _freeze_json(item) for name, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _thaw_json(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_constant(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("code constants must be finite")
        return {"float_hex": value.hex()}
    if isinstance(value, bytes):
        return {"bytes_b64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return {"tuple": [_canonical_constant(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_canonical_constant(item) for item in value]
        return {"frozenset": sorted(items, key=_canonical_json_bytes)}
    if isinstance(value, types.CodeType):
        return {"code": _code_payload(value)}
    if value is Ellipsis:
        return {"ellipsis": True}
    raise ValueError(f"unsupported code constant type: {type(value).__qualname__}")


def _code_payload(code: types.CodeType) -> dict[str, Any]:
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code_b64": base64.b64encode(code.co_code).decode("ascii"),
        "consts": [_canonical_constant(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _code_digest(code: types.CodeType) -> str:
    return _sha256(_canonical_json_bytes(_code_payload(code)))


def _find_compiled_code(module_code: types.CodeType, qualname: str) -> types.CodeType:
    pending = [module_code]
    matches: list[types.CodeType] = []
    while pending:
        current = pending.pop()
        if current.co_qualname == qualname:
            matches.append(current)
        pending.extend(
            item for item in current.co_consts if isinstance(item, types.CodeType)
        )
    if len(matches) != 1:
        raise ValueError(
            f"loaded callable {qualname!r} cannot be tied uniquely to current source"
        )
    return matches[0]


def _underlying_python_function(value: Any) -> types.FunctionType:
    candidate = getattr(value, "python_function", value)
    if not isinstance(candidate, types.FunctionType):
        raise ValueError("registered callable must expose an inspectable Python function")
    if "<locals>" in candidate.__qualname__ or candidate.__name__ == "<lambda>":
        raise ValueError("local functions and lambdas cannot receive route identity")
    return candidate


def _resolve_symbol(symbol: str) -> Any:
    module_name, separator, qualname = symbol.partition(":")
    if not separator or not module_name or not qualname:
        raise ValueError(f"invalid registered symbol: {symbol!r}")
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    return value


def _symbol_for(value: Any) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise ValueError("dependency has no stable module-qualified symbol")
    return f"{module}:{qualname}"


def _owned_module(module_name: str, roots: Sequence[str]) -> bool:
    return any(
        module_name == root.rstrip(".") or module_name.startswith(root)
        for root in roots
    )


def _root_module(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def _function_auxiliary_payload(function: types.FunctionType) -> dict[str, Any]:
    if function.__closure__:
        raise ValueError(
            f"registered callable {_symbol_for(function)} cannot close over runtime state"
        )
    return {
        "defaults": _canonical_constant(function.__defaults__),
        "kwdefaults": {
            str(name): _canonical_constant(value)
            for name, value in sorted((function.__kwdefaults__ or {}).items())
        },
    }


def _wrapper_payload(value: Any, role_spec: "_CallableRoleSpec") -> dict[str, Any]:
    if role_spec.wrapper_kind == "python_function":
        if getattr(value, "python_function", None) is not None:
            raise ValueError(f"{role_spec.role} must be an undecorated Python function")
        return {"wrapper_kind": "python_function"}
    if role_spec.wrapper_kind != "tensorflow_function":
        raise ValueError(f"unsupported wrapper kind: {role_spec.wrapper_kind}")
    if getattr(value, "python_function", None) is None:
        raise ValueError(f"{role_spec.role} must be a TensorFlow function wrapper")
    wrapper_type = type(value)
    if not str(wrapper_type.__module__).startswith("tensorflow."):
        raise ValueError(f"{role_spec.role} has a non-TensorFlow function wrapper")
    jit_compile = getattr(value, "_jit_compile", None)
    if jit_compile is not role_spec.jit_compile:
        raise ValueError(f"{role_spec.role} jit_compile does not match the route spec")
    input_signature = getattr(value, "input_signature", None)
    if input_signature is not None:
        signature: list[dict[str, Any]] = []
        for item in input_signature:
            shape = getattr(item, "shape", None)
            dtype = getattr(item, "dtype", None)
            if shape is None or dtype is None:
                raise ValueError("TensorFlow input_signature contains an unknown TypeSpec")
            signature.append(
                {
                    "type": f"{type(item).__module__}.{type(item).__qualname__}",
                    "shape": [None if dim is None else int(dim) for dim in shape],
                    "dtype": str(getattr(dtype, "name", dtype)),
                    "name": getattr(item, "name", None),
                }
            )
    else:
        signature = []
    return {
        "wrapper_kind": "tensorflow_function",
        "wrapper_type": f"{wrapper_type.__module__}.{wrapper_type.__qualname__}",
        "jit_compile": jit_compile,
        "autograph": bool(getattr(value, "_autograph", True)),
        "reduce_retracing": bool(getattr(value, "_reduce_retracing", False)),
        "input_signature": signature,
    }


def _global_value_record(name: str, value: Any, roles: set[str]) -> dict[str, Any]:
    if (
        isinstance(value, (np.ndarray, np.generic))
        or hasattr(value, "numpy")
    ):
        tensor, _ = _tensor_record(name, value)
        payload: Any = {"tensor": tensor}
    elif is_dataclass(value) and not isinstance(value, type):
        field_payload = []
        for item in fields(value):
            field_value = getattr(value, item.name)
            if (
                isinstance(field_value, (np.ndarray, np.generic))
                or hasattr(field_value, "numpy")
            ):
                record, _ = _tensor_record(item.name, field_value)
                encoded: Any = {"tensor": record}
            else:
                encoded = {"constant": _canonical_constant(field_value)}
            field_payload.append({"name": item.name, "payload": encoded})
        payload = {
            "dataclass_type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": field_payload,
        }
    elif hasattr(value, "name") and type(value).__module__.startswith("tensorflow"):
        payload = {
            "tensorflow_named_value": {
                "type": f"{type(value).__module__}.{type(value).__qualname__}",
                "name": str(value.name),
            }
        }
    else:
        payload = {"constant": _canonical_constant(value)}
    return {
        "global_name": name,
        "used_by_roles": sorted(roles),
        "payload": payload,
        "payload_sha256": _sha256(_canonical_json_bytes(payload)),
    }


def _is_serializable_global_value(value: Any) -> bool:
    return (
        value is None
        or isinstance(
            value,
            (bool, int, float, str, bytes, tuple, frozenset, np.ndarray, np.generic),
        )
        or hasattr(value, "numpy")
        or (is_dataclass(value) and not isinstance(value, type))
        or (
            hasattr(value, "name")
            and type(value).__module__.startswith("tensorflow")
        )
    )


def _tensor_record(name: str, value: Any) -> tuple[dict[str, Any], np.ndarray]:
    raw = value.numpy() if hasattr(value, "numpy") else value
    try:
        array = np.asarray(raw)
    except Exception as error:
        raise ValueError(f"prepared input {name} is not tensor-serializable") from error
    if array.dtype.hasobject:
        raise ValueError(f"prepared input {name} cannot use object dtype")
    # np.ascontiguousarray promotes scalars to shape [1], which corrupts the
    # prepared-input rank encoded in the identity.
    array = np.array(array, copy=True, order="C", subok=False)
    data = array.tobytes(order="C")
    record = {
        "name": name,
        "dtype": array.dtype.name,
        "dtype_encoding": array.dtype.str,
        "rank": array.ndim,
        "shape": list(array.shape),
        "byte_length": len(data),
        "value_sha256": _sha256(data),
    }
    return record, array


@dataclass(frozen=True)
class _PreparedFieldSpec:
    name: str
    semantic_role: str
    allowed_dtype_names: tuple[str, ...]
    shape: tuple[int | str, ...]
    finite_required: bool = True
    strictly_positive: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.semantic_role:
            raise ValueError("prepared field name and semantic_role must be nonempty")
        if not self.allowed_dtype_names or any(not item for item in self.allowed_dtype_names):
            raise ValueError("prepared field must declare allowed dtypes")
        if any(not isinstance(item, (int, str)) for item in self.shape):
            raise ValueError("prepared field shape entries must be integers or symbols")
        if any(isinstance(item, int) and item < 0 for item in self.shape):
            raise ValueError("prepared field fixed dimensions must be nonnegative")


@dataclass(frozen=True)
class _CallableRoleSpec:
    role: str
    symbol: str
    wrapper_kind: str = "python_function"
    jit_compile: bool | None = None

    def __post_init__(self) -> None:
        if not self.role or not self.symbol:
            raise ValueError("callable role and symbol must be nonempty")
        if self.wrapper_kind not in {"python_function", "tensorflow_function"}:
            raise ValueError("callable wrapper_kind is unsupported")
        if self.wrapper_kind == "python_function" and self.jit_compile is not None:
            raise ValueError("plain Python callable cannot declare jit_compile")
        if self.wrapper_kind == "tensorflow_function" and not isinstance(
            self.jit_compile, bool
        ):
            raise ValueError("TensorFlow callable must declare boolean jit_compile")


@dataclass(frozen=True)
class _RouteSpecification:
    route_specification_id: str
    row_id: str
    target_scalar: str
    target_output_tensor_field: str
    theta_coordinate_system: str
    parameter_names: tuple[str, ...]
    residual_design_id: str
    ridge_policy_id: str
    prepared_fields: tuple[_PreparedFieldSpec, ...]
    callable_roles: tuple[_CallableRoleSpec, ...]
    owned_dependency_symbols: tuple[str, ...] = ()
    allowed_external_roots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        text_fields = (
            self.route_specification_id,
            self.row_id,
            self.target_scalar,
            self.target_output_tensor_field,
            self.theta_coordinate_system,
            self.residual_design_id,
            self.ridge_policy_id,
        )
        if any(not value for value in text_fields):
            raise ValueError("route specification text fields must be nonempty")
        if not self.parameter_names or any(not item for item in self.parameter_names):
            raise ValueError("route specification parameter_names must be nonempty")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("route specification parameter_names must be unique")
        field_names = [item.name for item in self.prepared_fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("prepared field names must be unique")
        roles = [item.role for item in self.callable_roles]
        if set(roles) != {"reset", "value", "gradient"} or len(roles) != 3:
            raise ValueError("route specification requires reset, value, and gradient roles")
        if len(set(self.owned_dependency_symbols)) != len(self.owned_dependency_symbols):
            raise ValueError("owned dependency symbols must be unique")


@dataclass(frozen=True)
class _ExternalPrimitiveSpec:
    module_root: str
    distribution: str | None
    allowed_roles: tuple[str, ...]


@dataclass(frozen=True)
class _FactoryIssuedRouteIdentity:
    factory_scope: str
    route_specification_id: str
    row_id: str
    target_scalar: str
    target_output_tensor_field: str
    theta_coordinate_system: str
    parameter_names: tuple[str, ...]
    residual_design_id: str
    ridge_policy_id: str
    prepared_input_records: tuple[Mapping[str, Any], ...]
    prepared_input_sha256: str
    callable_records: tuple[Mapping[str, Any], ...]
    dependency_records: tuple[Mapping[str, Any], ...]
    global_value_records: tuple[Mapping[str, Any], ...]
    external_provenance: tuple[Mapping[str, Any], ...]
    source_dependency_closure_sha256: str
    identity_sha256: str
    _factory_seal: object = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTRACT_E_ROUTE_IDENTITY_SCHEMA_VERSION,
            "route_factory_id": CONTRACT_E_ROUTE_FACTORY_ID,
            "factory_scope": self.factory_scope,
            "route_specification_id": self.route_specification_id,
            "reset_contract_id": CONTRACT_E_RESET_CONTRACT_ID,
            "derivative_composition_id": CONTRACT_E_DERIVATIVE_COMPOSITION_ID,
            "row_normalization_policy_id": CONTRACT_E_ROW_NORMALIZATION_POLICY_ID,
            "row_id": self.row_id,
            "target_scalar": self.target_scalar,
            "target_output_tensor_field": self.target_output_tensor_field,
            "theta_coordinate_system": self.theta_coordinate_system,
            "parameter_names": list(self.parameter_names),
            "residual_design_id": self.residual_design_id,
            "ridge_policy_id": self.ridge_policy_id,
            "prepared_input_records": [
                _thaw_json(item) for item in self.prepared_input_records
            ],
            "prepared_input_sha256": self.prepared_input_sha256,
            "callable_records": [_thaw_json(item) for item in self.callable_records],
            "dependency_records": [_thaw_json(item) for item in self.dependency_records],
            "global_value_records": [
                _thaw_json(item) for item in self.global_value_records
            ],
            "external_provenance": [
                _thaw_json(item) for item in self.external_provenance
            ],
            "source_dependency_closure_sha256": self.source_dependency_closure_sha256,
            "identity_status": CONTRACT_E_PHASE2_CANDIDATE_STATUS,
            "admitted": False,
            "identity_sha256": self.identity_sha256,
        }


def _require_factory_identity(value: Any) -> _FactoryIssuedRouteIdentity:
    if not isinstance(value, _FactoryIssuedRouteIdentity):
        raise TypeError("route identity must be issued by the Contract E factory")
    issuance = _AUTHORIZED_FACTORY_SEALS.get(value._factory_seal)
    if issuance is None:
        raise ValueError("route identity has no authorized factory issuance seal")
    registered_digest, _factory_token = issuance
    expected = dict(value.to_dict())
    claimed = expected.pop("identity_sha256")
    if _sha256(_canonical_json_bytes(expected)) != claimed:
        raise ValueError("route identity digest does not match its fields")
    if registered_digest != claimed:
        raise ValueError("route identity differs from its registered issuance")
    return value


def _require_production_factory_identity(
    value: Any,
) -> _FactoryIssuedRouteIdentity:
    identity = _require_factory_identity(value)
    issuance = _AUTHORIZED_FACTORY_SEALS[identity._factory_seal]
    if issuance[1] is not _PRODUCTION_FACTORY._factory_token:
        raise ValueError(
            "public artifact builders require issuance by the repository "
            "production factory instance"
        )
    return identity


class _ContractERouteIdentityFactory:
    __slots__ = (
        "_route_specifications",
        "_owned_module_roots",
        "_external",
        "_factory_scope",
        "_factory_token",
        "_sealed",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("Contract E route factory is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        route_specifications: Sequence[_RouteSpecification],
        owned_module_roots: tuple[str, ...],
        external_primitive_specs: Sequence[_ExternalPrimitiveSpec],
        factory_scope: str,
        _construction_key: object,
    ) -> None:
        if _construction_key is not _FACTORY_CONSTRUCTION_KEY:
            raise TypeError("Contract E route factories are repository-owned")
        if not owned_module_roots:
            raise ValueError("factory requires at least one owned module root")
        if factory_scope not in {"production", "phase2_test_candidate"}:
            raise ValueError("factory_scope is unsupported")
        route_specifications = tuple(route_specifications)
        specifications = {
            item.route_specification_id: item for item in route_specifications
        }
        if len(specifications) != len(route_specifications):
            raise ValueError("route specification IDs must be unique")
        external_primitive_specs = tuple(external_primitive_specs)
        external = {item.module_root: item for item in external_primitive_specs}
        if len(external) != len(external_primitive_specs):
            raise ValueError("external primitive roots must be unique")
        self._route_specifications = MappingProxyType(specifications)
        self._owned_module_roots = tuple(owned_module_roots)
        self._external = MappingProxyType(external)
        self._factory_scope = factory_scope
        self._factory_token = object()
        self._sealed = True

    def _validate_prepared_inputs(
        self,
        specification: _RouteSpecification,
        prepared_inputs: Mapping[str, Any],
    ) -> tuple[tuple[Mapping[str, Any], ...], str]:
        if not isinstance(prepared_inputs, Mapping):
            raise TypeError("prepared_inputs must be a mapping of actual values")
        expected_names = {item.name for item in specification.prepared_fields}
        actual_names = set(prepared_inputs)
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing or extra:
            raise ValueError(
                f"prepared input field mismatch: missing={missing}, extra={extra}"
            )
        symbols: dict[str, int] = {}
        records: list[Mapping[str, Any]] = []
        for field_spec in specification.prepared_fields:
            record, array = _tensor_record(field_spec.name, prepared_inputs[field_spec.name])
            if record["dtype"] not in field_spec.allowed_dtype_names:
                raise ValueError(
                    f"prepared input {field_spec.name} has forbidden dtype {record['dtype']}"
                )
            if record["rank"] != len(field_spec.shape):
                raise ValueError(f"prepared input {field_spec.name} has wrong rank")
            for declared, actual in zip(field_spec.shape, record["shape"], strict=True):
                if isinstance(declared, int) and declared != actual:
                    raise ValueError(f"prepared input {field_spec.name} has wrong shape")
                if isinstance(declared, str):
                    previous = symbols.setdefault(declared, actual)
                    if previous != actual:
                        raise ValueError(
                            f"prepared input {field_spec.name} violates shape symbol {declared}"
                        )
            if field_spec.finite_required and not bool(np.all(np.isfinite(array))):
                raise ValueError(f"prepared input {field_spec.name} must be finite")
            if field_spec.strictly_positive and not bool(np.all(array > 0)):
                raise ValueError(
                    f"prepared input {field_spec.name} must be strictly positive"
                )
            enriched = dict(record)
            enriched["semantic_role"] = field_spec.semantic_role
            records.append(enriched)
        records.sort(key=lambda item: str(item["name"]))
        digest = _sha256(_canonical_json_bytes(records))
        return tuple(records), digest

    def _binding_record(
        self,
        role: str,
        symbol: str,
        value: Any,
        *,
        role_spec: _CallableRoleSpec | None = None,
    ) -> Mapping[str, Any]:
        resolved = _resolve_symbol(symbol)
        if value is not resolved:
            raise ValueError(f"{role} callable is not the exact registered symbol {symbol}")
        function = _underlying_python_function(value)
        expected_module, _, expected_qualname = symbol.partition(":")
        if (
            function.__module__ != expected_module
            or function.__qualname__ != expected_qualname
        ):
            raise ValueError(f"{role} callable was monkeypatched over its registered symbol")
        if not _owned_module(function.__module__, self._owned_module_roots):
            raise ValueError(f"{role} callable is outside the owned source roots")
        source_path_text = inspect.getsourcefile(function)
        if not source_path_text:
            raise ValueError(f"{role} callable has no source file")
        source_path = Path(source_path_text).resolve()
        try:
            relative_path = source_path.relative_to(_REPOSITORY_ROOT).as_posix()
        except ValueError as error:
            raise ValueError(f"{role} callable source is outside the repository") from error
        source_bytes = source_path.read_bytes()
        source_text = source_bytes.decode("utf-8")
        compiled = compile(source_text, str(source_path), "exec")
        source_code = _find_compiled_code(compiled, function.__qualname__)
        loaded_digest = _code_digest(function.__code__)
        source_digest = _code_digest(source_code)
        if loaded_digest != source_digest:
            raise ValueError(
                f"{role} loaded callable code does not match current inspected source"
            )
        source_segment = inspect.getsource(function).encode("utf-8")
        effective_role_spec = role_spec or _CallableRoleSpec(role=role, symbol=symbol)
        return {
            "role": role,
            "symbol": symbol,
            "module": function.__module__,
            "qualname": function.__qualname__,
            "source_file": relative_path,
            "source_file_sha256": _sha256(source_bytes),
            "source_segment_sha256": _sha256(source_segment),
            "loaded_code_sha256": loaded_digest,
            "function_auxiliary": _function_auxiliary_payload(function),
            "wrapper": _wrapper_payload(value, effective_role_spec),
        }

    def _scan_dependencies(
        self,
        root_functions: Mapping[str, types.FunctionType],
    ) -> tuple[dict[str, set[str]], dict[str, set[str]], tuple[Mapping[str, Any], ...]]:
        owned_symbols: dict[str, set[str]] = {}
        external_symbols: dict[str, set[str]] = {}
        global_values: dict[str, tuple[Any, set[str]]] = {}
        pending = [(role, function) for role, function in root_functions.items()]
        visited: set[tuple[int, str]] = set()
        while pending:
            role, function = pending.pop()
            visit_key = (id(function), role)
            if visit_key in visited:
                continue
            visited.add(visit_key)
            global_names: set[str] = set()
            code_pending = [function.__code__]
            while code_pending:
                code = code_pending.pop()
                global_names.update(
                    instruction.argval
                    for instruction in dis.get_instructions(code)
                    if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
                    and isinstance(instruction.argval, str)
                )
                code_pending.extend(
                    item for item in code.co_consts if isinstance(item, types.CodeType)
                )
            for name in sorted(global_names):
                if name not in function.__globals__:
                    if hasattr(builtins, name):
                        continue
                    raise ValueError(
                        f"unresolved global dependency {name!r} in {_symbol_for(function)}"
                    )
                dependency = function.__globals__[name]
                if isinstance(dependency, types.ModuleType):
                    module_name = dependency.__name__
                    if _owned_module(module_name, self._owned_module_roots):
                        raise ValueError(
                            "owned module globals are ambiguous dependencies; "
                            "import and register the exact symbol"
                        )
                    external_symbols.setdefault(module_name, set()).add(role)
                    continue
                if _is_serializable_global_value(dependency):
                    existing = global_values.get(name)
                    if existing is None:
                        global_values[name] = (dependency, {role})
                    elif existing[0] is not dependency and existing[0] != dependency:
                        raise ValueError(
                            f"global dependency {name!r} changed during scan"
                        )
                    else:
                        existing[1].add(role)
                    continue
                module_name = getattr(dependency, "__module__", None)
                if not isinstance(module_name, str):
                    continue
                if _owned_module(module_name, self._owned_module_roots):
                    symbol = _symbol_for(dependency)
                    owned_symbols.setdefault(symbol, set()).add(role)
                    pending.append((role, _underlying_python_function(dependency)))
                elif module_name != "builtins":
                    external_symbols.setdefault(_symbol_for(dependency), set()).add(role)
                else:
                    existing = global_values.get(name)
                    if existing is None:
                        global_values[name] = (dependency, {role})
                    elif existing[0] != dependency:
                        raise ValueError(f"global dependency {name!r} changed during scan")
                    else:
                        existing[1].add(role)
            for name in sorted(global_names):
                if name not in function.__globals__ or hasattr(builtins, name):
                    continue
                dependency = function.__globals__[name]
                if isinstance(dependency, types.ModuleType) or getattr(
                    dependency, "__module__", None
                ) is not None:
                    continue
                existing = global_values.get(name)
                if existing is None:
                    global_values[name] = (dependency, {role})
                else:
                    existing[1].add(role)
        global_records = tuple(
            _global_value_record(name, value, roles)
            for name, (value, roles) in sorted(global_values.items())
        )
        return owned_symbols, external_symbols, global_records

    def _external_provenance(
        self,
        specification: _RouteSpecification,
        external_symbols: Mapping[str, set[str]],
    ) -> tuple[Mapping[str, Any], ...]:
        roots = {_root_module(symbol.partition(":")[0]) for symbol in external_symbols}
        allowed = set(specification.allowed_external_roots)
        unexpected = sorted(roots - allowed)
        unused = sorted(allowed - roots)
        if unexpected or unused:
            raise ValueError(
                f"external dependency mismatch: unexpected={unexpected}, unused={unused}"
            )
        records: list[Mapping[str, Any]] = []
        for root in sorted(roots):
            references = [
                {
                    "symbol": symbol,
                    "used_by_roles": sorted(roles),
                }
                for symbol, roles in sorted(external_symbols.items())
                if _root_module(symbol.partition(":")[0]) == root
            ]
            used_roles = {
                role for reference in references for role in reference["used_by_roles"]
            }
            if root in sys.stdlib_module_names:
                records.append(
                    {
                        "module_root": root,
                        "provenance_kind": "python_standard_library",
                        "version": platform.python_version(),
                        "references": references,
                    }
                )
                continue
            primitive = self._external.get(root)
            if primitive is None:
                raise ValueError(f"external primitive {root!r} is not allowlisted")
            if not used_roles.issubset(set(primitive.allowed_roles)):
                raise ValueError(f"external primitive {root!r} is used in a forbidden role")
            if primitive.distribution is None:
                raise ValueError(f"external primitive {root!r} has ambiguous provenance")
            try:
                version = importlib.metadata.version(primitive.distribution)
            except importlib.metadata.PackageNotFoundError as error:
                raise ValueError(
                    f"external primitive {root!r} distribution is unavailable"
                ) from error
            records.append(
                {
                    "module_root": root,
                    "provenance_kind": "installed_distribution",
                    "distribution": primitive.distribution,
                    "version": version,
                    "references": references,
                }
            )
        return tuple(records)

    def issue(
        self,
        *,
        route_specification_id: str,
        callables: Mapping[str, Any],
        prepared_inputs: Mapping[str, Any],
    ) -> _FactoryIssuedRouteIdentity:
        specification = self._route_specifications.get(route_specification_id)
        if specification is None:
            raise ValueError(
                f"unregistered Contract E route specification: {route_specification_id}"
            )
        expected_roles = {item.role for item in specification.callable_roles}
        if set(callables) != expected_roles:
            raise ValueError("callable roles must exactly match the route specification")
        prepared_records, prepared_digest = self._validate_prepared_inputs(
            specification, prepared_inputs
        )
        callable_records: list[Mapping[str, Any]] = []
        functions: dict[str, types.FunctionType] = {}
        for role_spec in sorted(specification.callable_roles, key=lambda item: item.role):
            value = callables[role_spec.role]
            callable_records.append(
                self._binding_record(
                    role_spec.role,
                    role_spec.symbol,
                    value,
                    role_spec=role_spec,
                )
            )
            functions[role_spec.role] = _underlying_python_function(value)
        discovered_owned, discovered_external, global_records = self._scan_dependencies(
            functions
        )
        for role_spec in specification.callable_roles:
            if role_spec.wrapper_kind == "tensorflow_function":
                wrapper = callables[role_spec.role]
                wrapper_symbol = (
                    f"{type(wrapper).__module__}:{type(wrapper).__qualname__}"
                )
                discovered_external.setdefault(wrapper_symbol, set()).add(
                    role_spec.role
                )
        declared_owned = set(specification.owned_dependency_symbols)
        if set(discovered_owned) != declared_owned:
            raise ValueError(
                "owned dependency closure mismatch: "
                f"discovered={sorted(discovered_owned)}, "
                f"declared={sorted(declared_owned)}"
            )
        dependency_records = tuple(
            {
                **self._binding_record(
                    "dependency", symbol, _resolve_symbol(symbol)
                ),
                "used_by_roles": sorted(discovered_owned[symbol]),
            }
            for symbol in sorted(declared_owned)
        )
        external_records = self._external_provenance(
            specification, discovered_external
        )
        closure_payload = {
            "callables": callable_records,
            "dependencies": [dict(item) for item in dependency_records],
            "global_values": [dict(item) for item in global_records],
            "external_provenance": [dict(item) for item in external_records],
        }
        closure_digest = _sha256(_canonical_json_bytes(closure_payload))
        identity_fields = {
            "schema_version": CONTRACT_E_ROUTE_IDENTITY_SCHEMA_VERSION,
            "route_factory_id": CONTRACT_E_ROUTE_FACTORY_ID,
            "factory_scope": self._factory_scope,
            "route_specification_id": specification.route_specification_id,
            "reset_contract_id": CONTRACT_E_RESET_CONTRACT_ID,
            "derivative_composition_id": CONTRACT_E_DERIVATIVE_COMPOSITION_ID,
            "row_normalization_policy_id": CONTRACT_E_ROW_NORMALIZATION_POLICY_ID,
            "row_id": specification.row_id,
            "target_scalar": specification.target_scalar,
            "target_output_tensor_field": specification.target_output_tensor_field,
            "theta_coordinate_system": specification.theta_coordinate_system,
            "parameter_names": list(specification.parameter_names),
            "residual_design_id": specification.residual_design_id,
            "ridge_policy_id": specification.ridge_policy_id,
            "prepared_input_records": [dict(item) for item in prepared_records],
            "prepared_input_sha256": prepared_digest,
            "callable_records": callable_records,
            "dependency_records": [dict(item) for item in dependency_records],
            "global_value_records": [dict(item) for item in global_records],
            "external_provenance": [dict(item) for item in external_records],
            "source_dependency_closure_sha256": closure_digest,
            "identity_status": CONTRACT_E_PHASE2_CANDIDATE_STATUS,
            "admitted": False,
        }
        identity_digest = _sha256(_canonical_json_bytes(identity_fields))
        issuance_seal = object()
        identity = _FactoryIssuedRouteIdentity(
            factory_scope=self._factory_scope,
            route_specification_id=specification.route_specification_id,
            row_id=specification.row_id,
            target_scalar=specification.target_scalar,
            target_output_tensor_field=specification.target_output_tensor_field,
            theta_coordinate_system=specification.theta_coordinate_system,
            parameter_names=specification.parameter_names,
            residual_design_id=specification.residual_design_id,
            ridge_policy_id=specification.ridge_policy_id,
            prepared_input_records=tuple(_freeze_json(item) for item in prepared_records),
            prepared_input_sha256=prepared_digest,
            callable_records=tuple(_freeze_json(item) for item in callable_records),
            dependency_records=tuple(_freeze_json(item) for item in dependency_records),
            global_value_records=tuple(_freeze_json(item) for item in global_records),
            external_provenance=tuple(_freeze_json(item) for item in external_records),
            source_dependency_closure_sha256=closure_digest,
            identity_sha256=identity_digest,
            _factory_seal=issuance_seal,
        )
        _AUTHORIZED_FACTORY_SEALS[issuance_seal] = (
            identity_digest,
            self._factory_token,
        )
        return identity


_REVIEWED_EXTERNAL_PRIMITIVES = (
    _ExternalPrimitiveSpec(
        module_root="tensorflow",
        distribution="tensorflow",
        allowed_roles=("reset", "value", "gradient"),
    ),
    _ExternalPrimitiveSpec(
        module_root="tensorflow_probability",
        distribution="tensorflow-probability",
        allowed_roles=("reset", "value", "gradient"),
    ),
)

_LATENT_SIR_ROUTE_SPECIFICATION_ID = "contract_e_chol_latent_preclip_sir_austria_v1"
_LATENT_SIR_VALUE_SCORE_SYMBOL = (
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:"
    "latent_sir_contract_e_canonical_value_and_score_tf"
)
_LATENT_SIR_TWO_NODE_ROUTE_SPECIFICATION_ID = (
    "contract_e_chol_latent_preclip_sir_two_node_v1"
)
_LATENT_SIR_TWO_NODE_VALUE_SCORE_SYMBOL = (
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:"
    "latent_sir_two_node_contract_e_value_and_score_tf"
)
_LATENT_SIR_RESET_SYMBOL = (
    "bayesfilter.highdim.ledh_contract_e_reset_tf:contract_e_chol_cloud_forward_tf"
)
_LATENT_SIR_OWNED_DEPENDENCY_SYMBOLS = (
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:_cholesky_jvp",
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:_components",
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:_flow_forward_and_jvp",
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:_gaussian_density_and_jvp",
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:_geometry_and_jvp",
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:_inverse_jvp",
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:_normalize_and_jvp",
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:_physical_state_and_tangent",
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:_rhs_and_tangent",
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:_transition_mean_and_tangent",
    "bayesfilter.highdim.ledh_contract_e_latent_sir_tf:latent_sir_contract_e_value_and_score_core",
    "bayesfilter.highdim.ledh_contract_e_reset_tf:_apply_rows",
    "bayesfilter.highdim.ledh_contract_e_reset_tf:_cholesky_jvp",
    "bayesfilter.highdim.ledh_contract_e_reset_tf:_contract_e_chol_cloud_forward_core",
    "bayesfilter.highdim.ledh_contract_e_reset_tf:_contract_e_chol_cloud_jvp_core",
    "bayesfilter.highdim.ledh_contract_e_reset_tf:_factor_condition_proxy",
    "bayesfilter.highdim.ledh_contract_e_reset_tf:_right_triangular_solve",
    "bayesfilter.highdim.ledh_contract_e_reset_tf:_sym",
    "bayesfilter.highdim.ledh_contract_e_reset_tf:_uniform_moments",
    "bayesfilter.highdim.ledh_contract_e_reset_tf:_uniform_moments_jvp",
    "bayesfilter.highdim.ledh_contract_e_reset_tf:_weighted_moments",
    "bayesfilter.highdim.ledh_contract_e_reset_tf:_weighted_moments_jvp",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_augmented_payload",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_augmented_payload_tangent",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_balanced_transport_jvp_core",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_balanced_transport_value_core",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_contract_e_streaming_forward_core",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_contract_e_streaming_jvp_core",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_marginal_roundoff_tolerance",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_row_quotient_forward_core",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_row_quotient_jvp_core",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_streaming_column_masses_from_potentials_core",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_streaming_marginal_diagnostics_core",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_streaming_row_quotient_forward_core",
    "bayesfilter.highdim.ledh_contract_e_streaming_tf:_streaming_row_quotient_jvp_core",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_epsilon_per_batch",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_filterflow_streaming_column_log_normalizer",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_filterflow_streaming_column_log_normalizer_jvp",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_filterflow_streaming_finite_sinkhorn_potentials_jvp_total",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_filterflow_streaming_finite_sinkhorn_potentials_total_vjp",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_filterflow_streaming_softmin",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_filterflow_streaming_softmin_jvp",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_filterflow_streaming_terminal_balance_potential",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_filterflow_streaming_terminal_balance_potential_jvp",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_filterflow_streaming_transport_from_potentials",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_filterflow_streaming_transport_from_potentials_jvp",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_half_pairwise_squared_cross_jvp",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_logaddexp",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_pairwise_squared_cross",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_slice_axis1_padded_2d",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_slice_axis1_padded_3d",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_slice_axis1_padded_4d",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_streaming_log_zero",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_validate_chunk_size",
    "experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf:_validate_manual_dense_finite_route_inputs",
)
_LATENT_SIR_ROUTE_SPECIFICATION = _RouteSpecification(
    route_specification_id=_LATENT_SIR_ROUTE_SPECIFICATION_ID,
    row_id="zhao_cui_sir_austria_latent_preclip",
    target_scalar="finite_contract_e_observed_data_log_likelihood",
    target_output_tensor_field="objective",
    theta_coordinate_system="log_kappa_log_nu_log_observation_scale",
    parameter_names=(
        "log_kappa_scale",
        "log_nu_scale",
        "log_obs_noise_scale",
    ),
    residual_design_id="fixed_centered_realized_residual_design_v1",
    ridge_policy_id="prepared_parameter_independent_ridge_v1",
    prepared_fields=(
        _PreparedFieldSpec(
            "observations", "fixed observed SIR data", ("float64",), ("T", "J")
        ),
        _PreparedFieldSpec(
            "initial_noise", "fixed initial particle noise", ("float64",), ("B", "N", "D")
        ),
        _PreparedFieldSpec(
            "transition_noise",
            "fixed transition particle noise",
            ("float64",),
            ("B", "Tm1", "N", "D"),
        ),
        _PreparedFieldSpec(
            "fixed_reset_mask",
            "fixed Contract E reset schedule",
            ("bool",),
            ("B", "T"),
            finite_required=False,
        ),
        _PreparedFieldSpec(
            "residual_design",
            "fixed realized centered residual design",
            ("float64",),
            ("B", "T", "N", "D"),
        ),
        _PreparedFieldSpec(
            "prepared_ridge",
            "fixed parameter-independent realized ridge",
            ("float64",),
            ("B", "T"),
            strictly_positive=True,
        ),
        _PreparedFieldSpec(
            "epsilon", "fixed terminal transport epsilon", ("float64",), (),
            strictly_positive=True,
        ),
        _PreparedFieldSpec(
            "scaling", "fixed annealing scaling", ("float64",), (),
            strictly_positive=True,
        ),
    ),
    callable_roles=(
        _CallableRoleSpec(
            "reset", _LATENT_SIR_RESET_SYMBOL,
            wrapper_kind="tensorflow_function", jit_compile=True,
        ),
        _CallableRoleSpec(
            "value", _LATENT_SIR_VALUE_SCORE_SYMBOL,
            wrapper_kind="tensorflow_function", jit_compile=True,
        ),
        _CallableRoleSpec(
            "gradient", _LATENT_SIR_VALUE_SCORE_SYMBOL,
            wrapper_kind="tensorflow_function", jit_compile=True,
        ),
    ),
    owned_dependency_symbols=_LATENT_SIR_OWNED_DEPENDENCY_SYMBOLS,
    allowed_external_roots=("tensorflow",),
)
_LATENT_SIR_TWO_NODE_ROUTE_SPECIFICATION = replace(
    _LATENT_SIR_ROUTE_SPECIFICATION,
    route_specification_id=_LATENT_SIR_TWO_NODE_ROUTE_SPECIFICATION_ID,
    row_id="zhao_cui_sir_two_node_spatial_latent_preclip",
    callable_roles=(
        _CallableRoleSpec(
            "reset", _LATENT_SIR_RESET_SYMBOL,
            wrapper_kind="tensorflow_function", jit_compile=True,
        ),
        _CallableRoleSpec(
            "value", _LATENT_SIR_TWO_NODE_VALUE_SCORE_SYMBOL,
            wrapper_kind="tensorflow_function", jit_compile=True,
        ),
        _CallableRoleSpec(
            "gradient", _LATENT_SIR_TWO_NODE_VALUE_SCORE_SYMBOL,
            wrapper_kind="tensorflow_function", jit_compile=True,
        ),
    ),
)

_PRODUCTION_FACTORY = _ContractERouteIdentityFactory(
    route_specifications=(
        _LATENT_SIR_ROUTE_SPECIFICATION,
        _LATENT_SIR_TWO_NODE_ROUTE_SPECIFICATION,
    ),
    owned_module_roots=("bayesfilter.", "experiments."),
    external_primitive_specs=_REVIEWED_EXTERNAL_PRIMITIVES,
    factory_scope="production",
    _construction_key=_FACTORY_CONSTRUCTION_KEY,
)


def issue_contract_e_route_identity(
    *,
    route_specification_id: str,
    callables: Mapping[str, Any],
    prepared_inputs: Mapping[str, Any],
) -> _FactoryIssuedRouteIdentity:
    """Issue only identities registered by the repository production factory."""

    return _PRODUCTION_FACTORY.issue(
        route_specification_id=route_specification_id,
        callables=callables,
        prepared_inputs=prepared_inputs,
    )


def issue_latent_sir_contract_e_route_identity(
    *, prepared_inputs: Mapping[str, Any]
) -> _FactoryIssuedRouteIdentity:
    """Issue the SIR identity without accepting caller-selected callables."""

    from bayesfilter.highdim.ledh_contract_e_latent_sir_tf import (
        latent_sir_contract_e_canonical_value_and_score_tf,
    )
    from bayesfilter.highdim.ledh_contract_e_reset_tf import (
        contract_e_chol_cloud_forward_tf,
    )

    return _PRODUCTION_FACTORY.issue(
        route_specification_id=_LATENT_SIR_ROUTE_SPECIFICATION_ID,
        callables={
            "reset": contract_e_chol_cloud_forward_tf,
            "value": latent_sir_contract_e_canonical_value_and_score_tf,
            "gradient": latent_sir_contract_e_canonical_value_and_score_tf,
        },
        prepared_inputs=prepared_inputs,
    )


def issue_latent_sir_two_node_contract_e_route_identity(
    *, prepared_inputs: Mapping[str, Any]
) -> _FactoryIssuedRouteIdentity:
    """Issue the coupled two-node SIR identity with repository callables."""

    from bayesfilter.highdim.ledh_contract_e_latent_sir_tf import (
        latent_sir_two_node_contract_e_value_and_score_tf,
    )
    from bayesfilter.highdim.ledh_contract_e_reset_tf import (
        contract_e_chol_cloud_forward_tf,
    )

    return _PRODUCTION_FACTORY.issue(
        route_specification_id=_LATENT_SIR_TWO_NODE_ROUTE_SPECIFICATION_ID,
        callables={
            "reset": contract_e_chol_cloud_forward_tf,
            "value": latent_sir_two_node_contract_e_value_and_score_tf,
            "gradient": latent_sir_two_node_contract_e_value_and_score_tf,
        },
        prepared_inputs=prepared_inputs,
    )


def _make_test_candidate_factory(
    *,
    route_specifications: Sequence[_RouteSpecification],
    owned_module_roots: tuple[str, ...],
    external_primitive_specs: Sequence[_ExternalPrimitiveSpec] = (),
) -> _ContractERouteIdentityFactory:
    """Create a sealed candidate factory for private schema tests only."""

    return _ContractERouteIdentityFactory(
        route_specifications=route_specifications,
        owned_module_roots=owned_module_roots,
        external_primitive_specs=external_primitive_specs,
        factory_scope="phase2_test_candidate",
        _construction_key=_FACTORY_CONSTRUCTION_KEY,
    )
