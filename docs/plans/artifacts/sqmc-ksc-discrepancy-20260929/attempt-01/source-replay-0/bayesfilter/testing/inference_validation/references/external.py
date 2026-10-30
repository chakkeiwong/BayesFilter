"""Versioned external posterior observations, never silently treated as exact.

Supports portable JSON samples exported from Stan, PyMC, R or posteriordb.
An adapter must declare the identical target/data/prior/coordinate identity.
No installation, execution or network access occurs while reading references.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..storage import read_json,file_hash


def load_reference(path,*,target_identity,quantity_names):
    path=Path(path)
    bundle=read_json(path)
    if bundle.get("schema")!="bayesfilter.external_posterior_reference.v1": raise ValueError("unsupported external reference")
    if bundle.get("target_identity")!=target_identity: raise ValueError("external target/data/prior/coordinate identity mismatch")
    if tuple(bundle.get("quantity_names",()))!=tuple(quantity_names): raise ValueError("external quantity order mismatch")
    for key in ("source","version","method","uncertainty","dependency_independence"):
        if not bundle.get(key): raise ValueError(f"external reference requires {key}")
    samples_path=path.parent/bundle["samples_file"]
    if file_hash(samples_path)!=bundle.get("samples_sha256"): raise ValueError("external samples checksum mismatch")
    samples=np.asarray(read_json(samples_path),dtype=float)
    if samples.ndim!=2 or samples.shape[1]!=len(quantity_names) or len(samples)<2 or not np.all(np.isfinite(samples)):
        raise ValueError("invalid external reference samples")
    return samples,bundle
