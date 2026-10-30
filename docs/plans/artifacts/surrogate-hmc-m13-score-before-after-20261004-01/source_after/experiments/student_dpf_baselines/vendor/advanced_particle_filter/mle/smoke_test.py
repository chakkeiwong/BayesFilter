"""
Smoke test for the Find-Mode-Then-Sample pipeline.

Runs the full 3-phase pipeline with TINY settings to verify plumbing.
Expect nonsense numbers; we're only checking that nothing throws.

Usage:
    python -m advanced_particle_filter.mle.smoke_test
"""

from .run_mle_laplace import main


def smoke():
    print("\n" + "#" * 74)
    print("#  SMOKE TEST  (tiny config; DO NOT interpret numbers)")
    print("#" * 74)

    results = main(
        # data
        T=30,
        data_seed=0,
        # DPF small for speed
        n_particles=100,
        # Phase 1: very short
        B_restart=2,
        adam_steps=30,
        adam_lr=0.01,
        adam_n_mc=2,
        # Phase 2: minimal seeds
        laplace_eps=0.05,
        laplace_n_seeds=3,
        laplace_chunk_size=8,
        # Phase 3: very short
        run_phase3=True,
        hmc_B_chain=2,
        hmc_burnin=10,
        hmc_samples=20,
        hmc_leapfrog=3,
        hmc_step=0.3,
        hmc_n_mc=1,
        verbose=True,
    )

    # Basic shape checks
    assert results['adam'].z_hat.shape == (9,), "z_hat wrong shape"
    assert results['laplace'].Sigma_laplace.shape == (9, 9), "Sigma wrong shape"
    assert results['laplace'].L_chol.shape == (9, 9), "L_chol wrong shape"
    assert results['whitened'].samples_z.shape == (20, 2, 9), \
        "samples_z wrong shape"
    assert results['whitened'].samples_z_white.shape == (20, 2, 9), \
        "samples_z_white wrong shape"

    print("\n" + "#" * 74)
    print("#  SMOKE TEST: all shape checks passed.")
    print("#" * 74)
    return results


if __name__ == "__main__":
    smoke()
