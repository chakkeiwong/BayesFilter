"""CPU-only diagnostic: interrupted RWM must reproduce the uninterrupted chain."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import TargetPotentialEvaluation
from bayesfilter.inference.surrogate_hmc_program import file_hash, ROOT
from bayesfilter.testing import surrogate_hmc_m13_rwm_resume as resume
from bayesfilter.testing.surrogate_hmc_m13_rwm_baseline import (
    random_walk_graph, archive, STARTS, SEED, PRIOR_SD,
)

@pytest.fixture
def interrupted(tmp_path, monkeypatch):
    dtype = tf.float64
    def endpoint(x):
        return TargetPotentialEvaluation(
            .5*tf.reduce_sum(x*x, -1)/PRIOR_SD**2,
            dict(value_valid=tf.reduce_all(tf.math.is_finite(x), -1)))
    previous = tmp_path/"previous"
    out = previous/"worker/rwm"
    out.mkdir(parents=True)
    (previous/"report.json").write_text(json.dumps(dict(status="JOB_CPU_BUDGET_LIMIT")))
    source = "bayesfilter/testing/surrogate_hmc_m13_rwm_baseline.py"
    (previous/"run-manifest.json").write_text(json.dumps(dict(source_hashes={source:file_hash(ROOT/source)})))
    seed = tf.constant(SEED)
    fold = lambda i: tf.random.experimental.stateless_fold_in(seed, i)
    starts = tf.constant(STARTS,dtype)*PRIOR_SD
    adaptive = random_walk_graph(endpoint,512,adapt=True)
    metric = adaptive(starts,tf.math.log(tf.constant(.5,dtype)),tf.eye(2,dtype=dtype)*PRIOR_SD,fold(1))
    cloud = tf.reshape(metric["states"][-256:],[-1,2])
    centered = cloud-tf.reduce_mean(cloud,0)
    empirical = tf.linalg.matmul(centered,centered,transpose_a=True)/tf.cast(tf.shape(cloud)[0]-1,dtype)
    covariance = .9*empirical+.1*PRIOR_SD**2*tf.eye(2,dtype=dtype)
    factor = tf.linalg.cholesky(covariance)
    step = adaptive(metric["final_state"],tf.constant(0.,dtype),factor,fold(2))
    fresh = random_walk_graph(endpoint,500,adapt=False)(starts,step["log_scale"],factor,fold(3))
    retained = random_walk_graph(endpoint,250,adapt=False)
    chunks=[]; state=fresh["final_state"]
    for index in range(8):
        chunk=retained(state,step["log_scale"],factor,fold(10+index))
        chunks.append(chunk); state=chunk["final_state"]
    for name,chunk in (("metric-warmup",metric),("step-warmup",step),
                       ("fresh-warmup",fresh),("retained-00",chunks[0])):
        archive(out,name,chunk)
    coverage=tmp_path/"coverage.json"
    coverage.write_text(json.dumps(dict(qualified=True,reference_sha256="gaussian")))
    monkeypatch.setattr(resume,"load_reference",lambda p: (dict(reference_sha256="gaussian",observation_checksum="fixture"),None,None))
    monkeypatch.setattr(resume,"_load",lambda: (None,SimpleNamespace(observation_checksum="fixture"),None))
    monkeypatch.setattr(resume.mechanics,"endpoint_components",lambda f: (None,endpoint))
    monkeypatch.setattr(resume,"batched_endpoint",lambda f:f)
    monkeypatch.setattr(resume,"assess",lambda p,*a,**k: dict(status="EQUIVALENT",draws=int(p.shape[1])))
    return previous,coverage,chunks,covariance

def test_resume_matches_uninterrupted_chain_and_metric(interrupted,tmp_path):
    previous,coverage,expected,covariance=interrupted
    output=tmp_path/"output"
    output.mkdir()
    (output/"command.log").write_text("watchdog bookkeeping")
    (output/"run-manifest.json").write_text("{}")
    report=resume.run_resume(output,previous,"unused",coverage)
    tf.debugging.assert_equal(tf.constant(report["covariance"],tf.float64),covariance)
    assert [x["draws"] for x in report["assessments"]]==[1000,2000]
    assert report["warmup_excluded"]
    for i,expected_chunk in enumerate(expected):
        actual=resume._load_chunk(output/"rwm"/f"retained-{i:02}",250)
        for key in ("states","proposals","accepted","log_scale","final_potential"):
            tf.debugging.assert_equal(actual[key],expected_chunk[key])

@pytest.mark.parametrize("defect",["continuity","scale","final_state","shape","status"])
def test_resume_rejects_corrupted_archive(interrupted,tmp_path,defect):
    previous,coverage,chunks,_=interrupted
    directory=previous/"worker/rwm/retained-00"
    if defect=="status":
        (previous/"report.json").write_text(json.dumps(dict(status="COMPLETED")))
    else:
        key={"continuity":"initial_state","scale":"log_scale","final_state":"final_state","shape":"states"}[defect]
        value=chunks[0][key]
        value=value[:2] if defect=="shape" else value+tf.cast(.1,value.dtype)
        tf.io.write_file(str(directory/(key+".tfbin")),tf.io.serialize_tensor(value))
    with pytest.raises((ValueError,tf.errors.InvalidArgumentError)):
        resume.run_resume(tmp_path/"output",previous,"unused",coverage)
