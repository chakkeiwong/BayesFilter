"""G2.3 warmup budget and shrinkage sweep for 1.01 threshold."""
import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.hardbound import model_tf, joint_target_tf
from bayesfilter.hardbound.windowed_dense_mass_adaptation import (
    run_windowed_dense_mass_adaptation,
)

RNG = np.random.RandomState(20260826)
jt = joint_target_tf


@pytest.mark.hmc
@pytest.mark.parametrize("warmup,shrinkage", [
    (4000, 0.03),
    (4000, 0.01),
    (4000, 0.001),
])
def test_g2_3_warmup_shrinkage_sweep(warmup, shrinkage):
    """Sweep warmup budget and shrinkage λ to reach R-hat < 1.01."""
    fix = model_tf.FIXTURE
    T = 40
    sim = model_tf.simulate(
        tf.constant(fix.theta_bar_truth, tf.float64),
        tf.constant(fix.noise_scale_truth, tf.float64),
        horizon=T, seed=20260821, target_id="mf_c1_k40_hardmax")
    y_tf = sim["observations"]
    truth = np.array(list(fix.theta_bar_truth) + [np.log(5e-4)] * 3)

    nc = 4
    raw_truth = jt.raw_from_theta(tf.constant(truth[None, :], tf.float64))
    init = [
        tf.constant(raw_truth.numpy() + RNG.normal(size=(nc, 9)) * 0.05,
                    tf.float64),
        tf.constant(RNG.normal(size=(nc, 8)) * 0.1, tf.float64),
        tf.constant(RNG.normal(size=(nc, T, 8)) * 0.1, tf.float64),
    ]

    def lp(theta_raw, x0_raw, eta_raw):
        return jt.joint_log_prob_raw_batched(y_tf, theta_raw, x0_raw, eta_raw,
                                             "mf_c1_k40_hardmax")

    ns = 3000
    out = run_windowed_dense_mass_adaptation(
        target_log_prob_fn=lp,
        initial_states=init,
        num_warmup_steps=warmup,
        num_samples=ns,
        initial_step_size=1e-2,
        target_accept_prob=0.70,  # Amendment A3: repository standard, unconditional
        seed=20260822,
        mass_shrinkage=shrinkage)

    n_total = nc * ns
    rhat = out["rhat"][0].numpy()
    ess = out["ess"][0].numpy()

    print(f"\n{'='*70}")
    print(f"warmup={warmup} shrinkage={shrinkage}")
    print('='*70)
    print(f"max R-hat {rhat.max():.4f}  min ESS {ess.min():.1f}")
    print(f"per-theta R-hat {np.array2string(rhat, precision=4)}")
    print(f"per-theta ESS   {np.array2string(ess, precision=1)}")
    print(f"sampling divergences {out['divergences']} / {n_total}")
    print(f"warmup divergences {out['warmup_divergences']}")

    print("\nSlow windows:")
    for w in out["window_diagnostics"]:
        if w["update_mass"]:
            print(f"  window {w['index']} [{w['start']:5d}:{w['end']:5d}] "
                  f"len={w['end']-w['start']:4d} cond={w['condition_number']:.3e} "
                  f"pooled={w['pooled_draws']:5d} step={w['step_size_after']:.3e}")

    passed = np.all(rhat < 1.01)
    print(f"\nResult: {'PASS' if passed else 'FAIL'} (threshold 1.01)")
    print('='*70)

    assert out["divergences"] <= 0.001 * n_total, out["divergences"]
    assert np.all(rhat < 1.01), (warmup, shrinkage, rhat, ess)
