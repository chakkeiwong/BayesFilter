"""Initialization-only repair for the M13 exploratory public tuning consumer."""
from dataclasses import replace
import json
from pathlib import Path
import tensorflow as tf
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash
from bayesfilter.testing.surrogate_hmc_m13_rwm_baseline import PRIOR_SD

PLAN = "docs/plans/surrogate-hmc-m13-initialization-repair-2026-09-27.md"
RWM = ROOT/"docs/plans/artifacts/surrogate-hmc-m13-assessment-rwm-20260926-06"
REFERENCE = ROOT/"docs/plans/artifacts/surrogate-hmc-m13-assessment-reference-20260926-02/worker/reference/reference.json"

def geometry_from_warmup(states):
    states = tf.ensure_shape(tf.convert_to_tensor(states,tf.float64),[512,4,2])
    tf.debugging.assert_all_finite(states,"invalid archived metric warmup")
    cloud = tf.reshape(states[-256:],[-1,2])
    center = tf.reduce_mean(cloud,0)
    centered = cloud-center
    empirical = tf.linalg.matmul(centered,centered,transpose_a=True)/tf.cast(tf.shape(cloud)[0]-1,tf.float64)
    covariance = .9*empirical+.1*PRIOR_SD**2*tf.eye(2,dtype=tf.float64)
    factor = tf.linalg.cholesky(covariance)
    signs = tf.constant([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],tf.float64)
    starts = center+tf.linalg.matmul(signs,factor,transpose_b=True)
    return starts,center,covariance

def initialization_config(config):
    parent=json.loads((RWM/"report.json").read_text())
    if parent.get("status")!="COMPLETED" or parent.get("source_unchanged") is not True:
        raise ValueError("RWM resume did not complete unchanged")
    report_path=RWM/"worker/rwm/report.json"
    report=json.loads(report_path.read_text())
    reference=json.loads(REFERENCE.read_text())
    if (report.get("status")!="M13_RWM_BASELINE_COMPLETE" or
        report.get("health_passed") is not True or report.get("warmup_excluded") is not True or
        report.get("reference_sha256")!=reference.get("reference_sha256") or
        report["target"]["data_seed"]!=1704):
        raise ValueError("wrong RWM initialization scope or unhealthy archive")
    record=report["archives"]["metric-warmup"]["states"]
    path=RWM/"worker/rwm/metric-warmup/states.tfbin"
    if file_hash(path)!=record["sha256"]:
        raise ValueError("RWM warmup checksum mismatch")
    states=tf.io.parse_tensor(tf.io.read_file(str(path)),tf.float64)
    starts,center,covariance=geometry_from_warmup(states)
    metadata=dict(plan=PLAN, source="discarded RWM metric warmup only",
        warmup_sha256=file_hash(path),rwm_report_sha256=file_hash(report_path),
        center=center.numpy().tolist(),covariance=covariance.numpy().tolist(),
        retained_draws_used=0,posterior_qualification=False)
    return replace(config,initial_chain_positions=tuple(tuple(row) for row in starts.numpy().tolist()),
        initialization_provenance="four covariance-scale starts from discarded same-target RWM metric warmup; no retained draws",
        fresh_seed=(20260927,1102)),metadata
