"""Analytic supplied-map fixtures for HMC validation, with no training claim.

The exact map composes the existing noncentered target chart with a frozen
affine scale. Partial maps reconstruct a smooth, bounded log-scale correction
through the supported dense-IAF codec. These fixtures do not choose runtime
maps or issue evidence that a transport learner can discover them.
"""
from __future__ import annotations

import math

from .designs import digest


def supplied_funnel_map(kind: str, *, scale: float = 3.) -> dict:
    """Return a base-target identity and a reconstructable frozen payload.

    Partial maps use s(v)=2*scale*tanh(a*v/(4*scale)), with a=1 or 1/2.
    The exact case has base coordinates (v,u); partial cases have (v,x).
    Every case represents the same model law after the base chart is applied.
    """
    if kind not in {"exact", "partial", "partial_half"}:
        raise ValueError("unknown supplied funnel map")
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    from bayesfilter.inference.neutra_artifacts import finalize_dense_iaf_neutra_artifact_payload
    from .targets import ValidationTarget

    target_id = "funnel_noncentered" if kind == "exact" else "funnel"
    target = ValidationTarget(target_id, {"scale": scale})
    saturation = 2. * scale
    strength = {"exact": None, "partial": 1., "partial_half": .5}[kind]
    construction = {"kind": kind, "scale": scale, "strength": strength,
                    "saturation": saturation, "training_performed": False}
    components = [{"component_id": "scale_v", "kind": "affine", "dim": 3,
                   "offset": [0., 0., 0.], "scale": [scale, 1., 1.]}]
    if strength is not None:
        components.append({
            "component_id": "partial_funnel", "kind": "dense_autoregressive_iaf",
            "dim": 3, "hidden_layers": [2], "activation": "tanh",
            "scale_transform": "identity", "s_max": 1.,
            "masks_policy": "legacy_degree_masks_v1", "dtype": "float64",
            "weights": [
                [[strength / (2. * saturation), 0.], [0., 0.], [0., 0.]],
                [[0., saturation, saturation, 0., 0., 0.], [0.] * 6],
            ],
            "biases": [[0., 0.], [0.] * 6],
        })
    payload = finalize_dense_iaf_neutra_artifact_payload({
        "schema": "bayesfilter.neutra.dense_iaf_frozen_transport.v1",
        "transport_id": "validation-supplied-funnel-" + kind,
        "dimension": 3, "target_signature": target.adapter_signature(),
        "log_jacobian_available": True,
        "component_order": [c["component_id"] for c in components],
        "components": components,
        # The historical schema requires this field. It identifies analytic
        # construction bytes here and is explicitly not a training receipt.
        "training_state_hash": "sha256:" + digest(construction),
        "construction": construction,
        "nonclaims": ["analytic validation fixture; no transport training",
                      "no learned-map quality or universal tuning guarantee"],
    })
    return {"target": target_id, "parameters": {"scale": scale},
            "construction": construction, "transport_payload": payload}
