"""TensorFlow validation adapters for two explicitly matched posteriordb laws.

These are optional consumer fixtures, not replacements for the public tuner.
Stan source anchors: models/stan/eight_schools_noncentered.stan:6-20 and
models/stan/blr.stan:7-16 at upstream commit 5545a1dd07ae297c36edecbcd82aa49097b4c385.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import zipfile

import tensorflow as tf

from .designs import digest

UPSTREAM_COMMIT = "5545a1dd07ae297c36edecbcd82aa49097b4c385"
CASES = {
    "eight_schools-eight_schools_noncentered": ("eight_schools_noncentered", "eight_schools"),
    "sblrc-blr": ("blr", "sblrc"),
}


def read_zip_json(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1 or not names[0].endswith(".json"):
            raise ValueError("expected one JSON member in reference archive")
        return json.loads(archive.read(names[0]))


def load_case(checkout, case):
    """Read exact model/data linkage and preserve reference chains separately."""
    model,data_name = CASES[case]
    root = Path(checkout)/"posterior_database"
    paths = {"posterior":root/"posteriors"/(case+".json"),
        "model":root/"models/stan"/(model+".stan"),
        "data":root/"data/data"/(data_name+".json.zip"),
        "reference":root/"reference_posteriors/draws/draws"/(case+".json.zip"),
        "reference_info":root/"reference_posteriors/draws/info"/(case+".info.json")}
    posterior = json.loads(paths["posterior"].read_text())
    if (posterior["model_name"],posterior["data_name"],posterior["reference_posterior_name"]) != (model,data_name,case):
        raise ValueError("posteriordb model/data/reference mismatch")
    return {"case":case,"model":model,"data":read_zip_json(paths["data"]),
        "reference_chains":read_zip_json(paths["reference"]),
        "reference_info":json.loads(paths["reference_info"].read_text()),
        "files":{name:{"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
                 for name,path in paths.items()}}


class PosteriordbTarget:
    def __init__(self, case, data, *, jit_compile=True):
        if case not in CASES:
            raise ValueError("unsupported matched posteriordb law")
        self.case = case
        self.data = json.loads(json.dumps(data,allow_nan=False))
        if case.startswith("eight_schools"):
            if data.get("J") != 8 or len(data.get("y",[])) != 8 or len(data.get("sigma",[])) != 8:
                raise ValueError("eight-schools fixture requires eight observations/errors")
            if any(not math.isfinite(v) for v in data["y"]) or any(not math.isfinite(v) or v <= 0 for v in data["sigma"]):
                raise ValueError("finite data and positive standard errors required")
            self.model_names = ("mu","tau",*(f"theta[{j}]" for j in range(1,9)))
            self.active_names = ("mu","log_tau",*(f"z[{j}]" for j in range(1,9)))
        else:
            if data.get("D") != 5 or data.get("N") != 100 or len(data.get("X",[])) != 100 or len(data.get("y",[])) != 100:
                raise ValueError("sblrc fixture requires its 100 by 5 design")
            if any(len(row) != 5 or any(not math.isfinite(v) for v in row) for row in data["X"]) or any(not math.isfinite(v) for v in data["y"]):
                raise ValueError("finite regression data required")
            self.model_names = (*(f"beta[{j}]" for j in range(1,6)),"sigma")
            self.active_names = (*self.model_names[:-1],"log_sigma")
        self.parameter_dim = len(self.model_names)
        self._batch = tf.function(self._batch_score,
            input_signature=[tf.TensorSpec([None,self.parameter_dim],tf.float64)],
            autograph=False,jit_compile=jit_compile)

    def adapter_signature(self):
        return digest({"law":"matched_posteriordb_target.v1","case":self.case,"data":self.data,
                       "upstream_commit":UPSTREAM_COMMIT,"active_coordinates":self.active_names})

    def value_score_capability(self):
        from bayesfilter.inference.posterior_adapter import ValueScoreCapability
        return ValueScoreCapability(value_score_authority="graph_native",xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,target_scope="inference_validation",
            runtime_backend="tensorflow",evidence_path=__file__,
            nonclaims=("matched validation fixture; reference uncertainty remains",))

    def parameter_names(self):
        return self.active_names

    def to_model(self,q):
        if self.case.startswith("eight_schools"):
            tau = tf.exp(q[...,1:2])
            return tf.concat([q[...,:1],tau,q[...,:1]+tau*q[...,2:]],axis=-1)
        return tf.concat([q[...,:-1],tf.exp(q[...,-1:])],axis=-1)

    def log_density(self,q):
        y = tf.constant(self.data["y"],tf.float64)
        if self.case.startswith("eight_schools"):
            mu,ell,z = q[...,0],q[...,1],q[...,2:]
            tau = tf.exp(ell)
            sigma = tf.constant(self.data["sigma"],tf.float64)
            residual = y-mu[...,None]-tau[...,None]*z
            # Normalizing constants independent of q are omitted, as in Stan's target.
            return (-.5*tf.square(mu/5.)-.5*tf.reduce_sum(z*z,axis=-1)
                -tf.nn.softplus(2*ell-math.log(25.))+ell
                -.5*tf.reduce_sum(tf.square(residual/sigma),axis=-1))
        beta,ell = q[...,:5],q[...,5]
        residual = y-tf.einsum("nd,...d->...n",tf.constant(self.data["X"],tf.float64),beta)
        return (-.5*tf.reduce_sum(tf.square(beta/10.),axis=-1)-.5*tf.exp(2*ell)/100.
            +(1-self.data["N"])*ell-.5*tf.reduce_sum(residual*residual,axis=-1)*tf.exp(-2*ell))

    def _batch_score(self,q):
        with tf.GradientTape() as tape:
            tape.watch(q)
            value = self.log_density(q)
        return value,tape.gradient(value,q)

    def log_prob_and_grad(self,position):
        q = tf.convert_to_tensor(position,tf.float64)
        scalar = q.shape.rank == 1
        value,score = self._batch(q[None] if scalar else q)
        return (value[0],score[0]) if scalar else (value,score)
