"""Legacy NeuTra transport-state import helpers.

These helpers convert reviewed legacy transport-state dictionaries into
BayesFilter dense-IAF payloads. They do not train NeuTra, run HMC, import
external model modules, or establish sampler/posterior validity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


LEGACY_DENSE_IAF_SCHEMA = "bayesfilter.neutra.dense_iaf_frozen_transport.v1"
LEGACY_NEUTRA_IMPORT_NONCLAIMS = (
    "legacy transport-state import only",
    "no NeuTra training claim",
    "no HMC tuning or sampling claim",
    "no posterior convergence claim",
    "no scientific validity claim",
    "no default policy change",
)


class InvalidLegacyNeuTraImport(ValueError):
    """Raised when a legacy transport-state payload is incomplete or unsafe."""


def build_dense_iaf_payload_from_legacy_training_state(
    training_state: Mapping[str, Any],
    *,
    transport_id: str,
    target_signature: str,
    training_state_hash: str | None = None,
    expected_dimension: int | None = None,
    component_id_prefix: str = "legacy",
    legacy_payload_references: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return a raw dense-IAF payload from one legacy training-state document.

    The returned mapping is intentionally *not* finalized. Call
    ``finalize_dense_iaf_neutra_artifact_payload`` and then
    ``load_frozen_neutra_artifact`` as the acceptance gates.
    """

    if not isinstance(training_state, Mapping):
        raise InvalidLegacyNeuTraImport("training_state must be a mapping")
    if "transport_state" not in training_state:
        raise InvalidLegacyNeuTraImport("training_state missing transport_state")
    payload = build_dense_iaf_payload_from_legacy_transport_state(
        training_state["transport_state"],
        transport_id=transport_id,
        target_signature=target_signature,
        training_state_hash=training_state_hash,
        expected_dimension=expected_dimension,
        component_id_prefix=component_id_prefix,
        legacy_payload_references=legacy_payload_references,
    )
    return payload


def build_dense_iaf_payload_from_legacy_transport_state(
    transport_state: Mapping[str, Any],
    *,
    transport_id: str,
    target_signature: str,
    training_state_hash: str | None = None,
    expected_dimension: int | None = None,
    component_id_prefix: str = "legacy",
    legacy_payload_references: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return a raw BayesFilter dense-IAF payload from legacy transport state.

    Supported legacy component kinds:

    - ``dense_autoregressive_iaf``
    - ``mixing_linear``
    - ``affine``

    Current constraint: dense-IAF legacy components must have ``s_max == 1.0``.
    This is required because the validated c603 bridge used that exact setting.
    """

    root = _require_mapping(transport_state, "transport_state")
    if _nonempty_text(root.get("type"), "transport_state type") != "composed":
        raise InvalidLegacyNeuTraImport("legacy root transport_state must be composed")
    children = _require_sequence(root.get("children"), "transport_state children")
    if not children:
        raise InvalidLegacyNeuTraImport("legacy root transport_state children must be nonempty")

    dimension = _infer_dimension(root, expected_dimension=expected_dimension)
    component_order: list[str] = []
    components: list[Mapping[str, Any]] = []
    for index, item in enumerate(children):
        child = _require_mapping(item, f"transport_state children[{index}]")
        child_type = _nonempty_text(child.get("type"), f"transport_state children[{index}] type")
        component_id = f"{component_id_prefix}_{index:02d}_{child_type}"
        component_order.append(component_id)
        components.append(
            _convert_legacy_component(
                child,
                component_id=component_id,
                dimension=dimension,
            )
        )

    payload: dict[str, Any] = {
        "schema": LEGACY_DENSE_IAF_SCHEMA,
        "transport_id": _nonempty_text(transport_id, "transport_id"),
        "dimension": dimension,
        "target_signature": _nonempty_text(target_signature, "target_signature"),
        "log_jacobian_available": True,
        "component_order": component_order,
        "components": components,
        "nonclaims": LEGACY_NEUTRA_IMPORT_NONCLAIMS,
    }
    if training_state_hash is not None:
        payload["training_state_hash"] = _nonempty_text(
            training_state_hash, "training_state_hash"
        )
    if legacy_payload_references is not None:
        payload["legacy_payload_references"] = _normalize_mapping(
            legacy_payload_references,
            "legacy_payload_references",
        )
    return payload


def _convert_legacy_component(
    component: Mapping[str, Any],
    *,
    component_id: str,
    dimension: int,
) -> Mapping[str, Any]:
    kind = _nonempty_text(component.get("type"), f"{component_id} type")
    if kind == "dense_autoregressive_iaf":
        return _convert_dense_iaf_component(
            component,
            component_id=component_id,
            dimension=dimension,
        )
    if kind == "mixing_linear":
        return _convert_mixing_linear_component(
            component,
            component_id=component_id,
            dimension=dimension,
        )
    if kind == "affine":
        return _convert_affine_component(
            component,
            component_id=component_id,
            dimension=dimension,
        )
    raise InvalidLegacyNeuTraImport(f"unsupported legacy component kind: {kind}")


def _convert_dense_iaf_component(
    component: Mapping[str, Any],
    *,
    component_id: str,
    dimension: int,
) -> Mapping[str, Any]:
    dim = _require_positive_int(component.get("dim"), f"{component_id} dim")
    if dim != dimension:
        raise InvalidLegacyNeuTraImport(
            f"{component_id} dim mismatch: expected {dimension}, got {dim}"
        )
    hidden_layers = tuple(
        _require_positive_int(item, f"{component_id} hidden_layers item")
        for item in _require_sequence(component.get("hidden_layers"), f"{component_id} hidden_layers")
    )
    activation = _nonempty_text(component.get("activation"), f"{component_id} activation")
    if activation not in {"elu", "tanh", "relu"}:
        raise InvalidLegacyNeuTraImport(f"{component_id} unsupported activation: {activation}")
    s_max = _require_finite_float(component.get("s_max"), f"{component_id} s_max")
    if abs(s_max - 1.0) > 1.0e-12:
        raise InvalidLegacyNeuTraImport(
            f"{component_id} legacy dense IAF requires s_max == 1.0"
        )
    weights_in = _require_sequence(component.get("weights"), f"{component_id} weights")
    biases_in = _require_sequence(component.get("biases"), f"{component_id} biases")
    layer_sizes = (dimension, *hidden_layers, 2 * dimension)
    if len(weights_in) != len(layer_sizes) - 1 or len(biases_in) != len(layer_sizes) - 1:
        raise InvalidLegacyNeuTraImport(f"{component_id} parameter length mismatch")
    weights = []
    biases = []
    for index, (weight_value, bias_value) in enumerate(zip(weights_in, biases_in)):
        weight = _finite_matrix(
            weight_value,
            f"{component_id} weights[{index}]",
            rows=layer_sizes[index],
            cols=layer_sizes[index + 1],
        )
        bias = _finite_vector(
            bias_value,
            f"{component_id} biases[{index}]",
            length=layer_sizes[index + 1],
        )
        weights.append(weight)
        biases.append(bias)
    return {
        "component_id": component_id,
        "kind": "dense_autoregressive_iaf",
        "dim": dimension,
        "hidden_layers": hidden_layers,
        "activation": activation,
        "s_max": s_max,
        "masks_policy": "legacy_degree_masks_v1",
        "dtype": "float64",
        "weights": tuple(weights),
        "biases": tuple(biases),
    }


def _convert_mixing_linear_component(
    component: Mapping[str, Any],
    *,
    component_id: str,
    dimension: int,
) -> Mapping[str, Any]:
    weight = _finite_matrix(
        component.get("W"),
        f"{component_id} W",
        rows=dimension,
        cols=dimension,
    )
    # Legacy dsge_hmc computes z @ W.T. BayesFilter mixing_linear computes values @ matrix.
    matrix = tuple(
        tuple(weight[row][col] for row in range(dimension))
        for col in range(dimension)
    )
    return {
        "component_id": component_id,
        "kind": "mixing_linear",
        "dim": dimension,
        "dtype": "float64",
        "matrix": matrix,
    }


def _convert_affine_component(
    component: Mapping[str, Any],
    *,
    component_id: str,
    dimension: int,
) -> Mapping[str, Any]:
    offset = _finite_vector(component.get("offset"), f"{component_id} offset", length=dimension)
    out: dict[str, Any] = {
        "component_id": component_id,
        "kind": "affine",
        "dim": dimension,
        "dtype": "float64",
        "offset": offset,
    }
    has_matrix = component.get("L_np") is not None
    has_scale = component.get("scale") is not None
    if has_matrix and has_scale:
        raise InvalidLegacyNeuTraImport(
            f"{component_id} affine payload must not contain both L_np and scale"
        )
    if has_matrix:
        out["L_np"] = _finite_matrix(
            component.get("L_np"),
            f"{component_id} L_np",
            rows=dimension,
            cols=dimension,
        )
        return out
    if has_scale:
        out["scale"] = _finite_vector(
            component.get("scale"),
            f"{component_id} scale",
            length=dimension,
        )
        return out
    raise InvalidLegacyNeuTraImport(f"{component_id} affine payload requires L_np or scale")


def _infer_dimension(
    root: Mapping[str, Any],
    *,
    expected_dimension: int | None,
) -> int:
    candidates: list[int] = []
    metadata = root.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("dim") is not None:
        candidates.append(_require_positive_int(metadata.get("dim"), "transport_state metadata dim"))
    for child in _require_sequence(root.get("children"), "transport_state children"):
        if isinstance(child, Mapping):
            child_dim = child.get("dim")
            if child_dim is not None:
                candidates.append(_require_positive_int(child_dim, "legacy child dim"))
            meta = child.get("metadata")
            if isinstance(meta, Mapping) and meta.get("dim") is not None:
                candidates.append(_require_positive_int(meta.get("dim"), "legacy child metadata dim"))
    if expected_dimension is not None:
        candidates.append(_require_positive_int(expected_dimension, "expected_dimension"))
    if not candidates:
        raise InvalidLegacyNeuTraImport("could not infer legacy transport dimension")
    dimension = candidates[0]
    for item in candidates[1:]:
        if item != dimension:
            raise InvalidLegacyNeuTraImport(
                f"legacy transport dimension mismatch: {dimension} vs {item}"
            )
    return dimension


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidLegacyNeuTraImport(f"{name} must be a mapping")
    return value


def _normalize_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidLegacyNeuTraImport(f"{name} must be a mapping")
    out: dict[str, Any] = {}
    for key, item in value.items():
        out[str(key)] = _normalize_json_value(item, f"{name}.{key}")
    return out


def _normalize_json_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return _normalize_mapping(value, name)
    if isinstance(value, (tuple, list)):
        return [_normalize_json_value(item, f"{name}[]") for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidLegacyNeuTraImport(f"{name} must be finite")
        return float(value)
    return str(value)


def _require_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (tuple, list)):
        raise InvalidLegacyNeuTraImport(f"{name} must be a sequence")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise InvalidLegacyNeuTraImport(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise InvalidLegacyNeuTraImport(f"{name} must be positive")
    return parsed


def _require_finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidLegacyNeuTraImport(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise InvalidLegacyNeuTraImport(f"{name} must be finite")
    return parsed


def _finite_vector(value: Any, name: str, *, length: int) -> tuple[float, ...]:
    items = _require_sequence(value, name)
    if len(items) != length:
        raise InvalidLegacyNeuTraImport(
            f"{name} length mismatch: expected {length}, got {len(items)}"
        )
    return tuple(_require_finite_float(item, f"{name}[{index}]") for index, item in enumerate(items))


def _finite_matrix(
    value: Any,
    name: str,
    *,
    rows: int,
    cols: int,
) -> tuple[tuple[float, ...], ...]:
    outer = _require_sequence(value, name)
    if len(outer) != rows:
        raise InvalidLegacyNeuTraImport(
            f"{name} row count mismatch: expected {rows}, got {len(outer)}"
        )
    out = []
    for row_index, row in enumerate(outer):
        values = _require_sequence(row, f"{name}[{row_index}]")
        if len(values) != cols:
            raise InvalidLegacyNeuTraImport(
                f"{name}[{row_index}] col count mismatch: expected {cols}, got {len(values)}"
            )
        out.append(
            tuple(
                _require_finite_float(item, f"{name}[{row_index}][{col_index}]")
                for col_index, item in enumerate(values)
            )
        )
    return tuple(out)


def _nonempty_text(value: Any, name: str) -> str:
    text = str(value)
    if not text.strip() or text == "None":
        raise InvalidLegacyNeuTraImport(f"{name} must be nonempty")
    return text
