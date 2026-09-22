"""Synthetic acceptance-policy interpretation; these are not HMC draws."""
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("CPU diagnostic requires hidden GPUs")
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
started = time.monotonic()
import tensorflow as tf
from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy, evaluate_hmc_acceptance_evidence

samples = tf.random.stateless_normal([64, 4, 2], (20260915, 101), dtype=tf.float64)
probabilities = tf.constant([.73, .75, .77, .79], tf.float64)
evidence = evaluate_hmc_acceptance_evidence(samples=samples,
    log_accept_ratio=tf.repeat(tf.math.log(probabilities)[None, :], 64, axis=0),
    is_accepted=tf.ones([64, 4], tf.bool), policy=HMCAcceptancePolicy())
result = {"role": "synthetic policy interpretation only; not coherent HMC transition evidence",
    "chain_means": evidence.chain_means, "pooled": evidence.pooled_mean,
    "interval": evidence.chain_mean_uncertainty_interval,
    "decision": evidence.acceptance_decision, "promotion_vetoes": evidence.candidate_promotion_vetoes,
    "wall_seconds": time.monotonic()-started}
Path(__file__).with_name("acceptance-check.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
