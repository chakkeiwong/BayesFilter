"""Simple versioned JSON and TensorFlow tensor records for local diagnostics."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def json_ready(value):
    if isinstance(value,dict): return {str(k):json_ready(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [json_ready(v) for v in value]
    if isinstance(value,float): return value if math.isfinite(value) else None
    if value is None or isinstance(value,(str,int,bool)): return value
    # Materialization is a reporting boundary only; no numerical computation here.
    if hasattr(value,"numpy"): return json_ready(value.numpy().tolist())
    if hasattr(value,"tolist"): return json_ready(value.tolist())
    raise TypeError(f"not serializable: {type(value).__name__}")


def write_json(path,value):
    from bayesfilter.runtime import atomic_write_json
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    atomic_write_json(path,json_ready(value))
    return path


def read_json(path):
    def reject_constant(x): raise ValueError(f"nonfinite JSON: {x}")
    return json.loads(Path(path).read_text(),parse_constant=reject_constant)


def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_tensor(path,tensor):
    import tensorflow as tf
    from bayesfilter.runtime.durable_tensor_checkpoint import durable_bytes
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tensor=tf.convert_to_tensor(tensor,tf.float64)
    raw=tf.io.serialize_tensor(tensor).numpy()
    if path.exists() and path.read_bytes()!=raw:
        raise ValueError("cannot replace a different numerical observation")
    durable_bytes(path,raw)
    write_json(str(path)+".json",{"sha256":file_hash(path),"dtype":"float64","shape":tensor.shape.as_list()})
    return path


def read_tensor(path):
    import tensorflow as tf
    path=Path(path); meta=read_json(str(path)+".json")
    if file_hash(path)!=meta["sha256"]: raise ValueError("tensor checksum mismatch")
    if meta["dtype"] != "float64": raise ValueError("unexpected tensor dtype")
    return tf.ensure_shape(tf.io.parse_tensor(path.read_bytes(),tf.float64),meta["shape"])
