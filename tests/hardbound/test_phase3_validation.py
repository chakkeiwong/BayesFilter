"""Phase 3 gates G3.1 (Geweke) and G3.2 (SBC) of the master program.

Marked extended+hmc: deliberate long runs, not part of the fast gate.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from scipy import stats

from bayesfilter.hardbound import validation_harness_tf as vh

ARTIFACT = "docs/plans/hardbound-kink-hmc-phase3-artifacts-2026-08-21.json"


@pytest.mark.extended
@pytest.mark.hmc
def test_g3_1_geweke_joint_distribution():
    t0 = time.time()
    res = vh.geweke_test(horizon=8, n_mc=4000, n_sc=4000,
                         transitions_per_step=25, seed=20260821)
    z = np.abs(res.z_scores)
    payload = {"geweke_z": dict(zip(res.names, res.z_scores.tolist())),
               "runtime_s": time.time() - t0}
    try:
        with open(ARTIFACT) as fh:
            existing = json.load(fh)
    except FileNotFoundError:
        existing = {}
    existing.update(payload)
    with open(ARTIFACT, "w") as fh:
        json.dump(existing, fh, indent=1)
    assert np.all(z < 4.0), dict(zip(res.names, res.z_scores))
    assert int((z > 3.0).sum()) < 2, dict(zip(res.names, res.z_scores))


@pytest.mark.extended
@pytest.mark.hmc
def test_g3_2_sbc_rank_uniformity():
    t0 = time.time()
    n_reps, n_post = 200, 99  # ranks in 0..99 -> 100 possible values
    ranks = vh.sbc(horizon=20, n_reps=n_reps, n_posterior=n_post,
                   warmup=600, seed=20260821)
    n_bins = 20
    edges = np.linspace(0, n_post + 1, n_bins + 1)
    pvals = []
    for j in range(9):
        counts, _ = np.histogram(ranks[:, j], bins=edges)
        chi2 = ((counts - n_reps / n_bins) ** 2 / (n_reps / n_bins)).sum()
        pvals.append(1.0 - stats.chi2.cdf(chi2, df=n_bins - 1))
    payload = {"sbc_pvals": pvals, "sbc_ranks": ranks.tolist(),
               "sbc_runtime_s": time.time() - t0}
    try:
        with open(ARTIFACT) as fh:
            existing = json.load(fh)
    except FileNotFoundError:
        existing = {}
    existing.update(payload)
    with open(ARTIFACT, "w") as fh:
        json.dump(existing, fh, indent=1)
    # Bonferroni across 9 parameters at family level 0.045
    assert min(pvals) > 0.005, pvals
