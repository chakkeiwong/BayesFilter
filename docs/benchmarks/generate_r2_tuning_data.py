"""Generate fresh tuning data for R2-TUNE campaigns (disjoint from claim partition).

CORRECTED 2026-08-28: Call each model's canonical simulation surface rather
than hand-coding approximations or perturbing frozen claim data.

Usage:
    python generate_r2_tuning_data.py --model Austria --seed 9006 --horizon 20 --replications 10

Outputs {model}_tuning_data.npz with observations, seeds, horizon, model_seed, SHA-256.
"""

import argparse
import hashlib
import numpy as np
import tensorflow as tf

DTYPE = tf.float64


def generate_linear2d(model_seed, horizon, replications):
    """Generate from 2D LGSSM test fixture."""
    import sys
    sys.path.insert(0, 'tests/highdim')
    from test_ledh_canonical_filter import _lgssm_model

    spec = _lgssm_model(101)  # Frozen claim data uses seed 101

    # Use different seed for tuning
    rng = np.random.default_rng(model_seed)
    dim = len(spec["initial_mean"])
    obs_dim = spec["obs_matrix"].shape[0]

    all_obs = []
    replication_seeds = []

    transition = spec["transition"]
    obs_matrix = spec["obs_matrix"]
    process_cov = spec["process_cov"]
    obs_cov = spec["obs_cov"]
    initial_mean = spec["initial_mean"]
    initial_cov = spec["initial_cov"]

    process_chol = np.linalg.cholesky(process_cov)
    obs_chol = np.linalg.cholesky(obs_cov)
    initial_chol = np.linalg.cholesky(initial_cov)

    for rep in range(replications):
        rep_seed = model_seed + 10000 + rep
        replication_seeds.append(rep_seed)
        rep_rng = np.random.default_rng(rep_seed)

        # Initial state
        x = initial_mean + rep_rng.standard_normal(dim) @ initial_chol.T

        obs_traj = []
        for t in range(horizon):
            # Transition
            x = transition @ x + rep_rng.standard_normal(dim) @ process_chol.T
            # Observation
            y = obs_matrix @ x + rep_rng.standard_normal(obs_dim) @ obs_chol.T
            obs_traj.append(y)

        all_obs.append(np.array(obs_traj))

    return np.array(all_obs), np.array(replication_seeds)


def generate_dlgssm(model_seed, horizon, replications):
    """Generate from diagonal LGSSM."""
    from bayesfilter.highdim.ledh_canonical_models_tf import diagonal_lgssm_canonical_model

    theta0_np = [0.9, 0.8, 0.7, 0.6, 0.8]
    theta0 = tf.constant(theta0_np, DTYPE)
    model, _ = diagonal_lgssm_canonical_model(theta0)

    rng = np.random.default_rng(model_seed)
    dim = 3
    obs_dim = 3

    all_obs = []
    replication_seeds = []

    phi = np.diag(theta0_np[:3])
    q = theta0_np[3]**2 * np.eye(3)
    r = theta0_np[4]**2 * np.eye(3)
    obs_matrix = np.array([[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]])

    process_chol = np.linalg.cholesky(q)
    obs_chol = np.linalg.cholesky(r)

    for rep in range(replications):
        rep_seed = model_seed + 10000 + rep
        replication_seeds.append(rep_seed)
        rep_rng = np.random.default_rng(rep_seed)

        x = rep_rng.standard_normal(dim)
        obs_traj = []
        for t in range(horizon):
            x = phi @ x + rep_rng.standard_normal(dim) @ process_chol.T
            y = obs_matrix @ x + rep_rng.standard_normal(obs_dim) @ obs_chol.T
            obs_traj.append(y)

        all_obs.append(np.array(obs_traj))

    return np.array(all_obs), np.array(replication_seeds)


def generate_predator_prey(model_seed, horizon, replications):
    """Generate from predator-prey model using its canonical .simulate() method."""
    from bayesfilter.highdim.models import p30_predator_prey_fixture_model

    model = p30_predator_prey_fixture_model()

    all_obs = []
    replication_seeds = []

    for rep in range(replications):
        rep_seed = model_seed + 10000 + rep
        replication_seeds.append(rep_seed)

        # Call the model's canonical .simulate() — RK4 integrator, 20 internal steps
        with tf.device("/CPU:0"):
            states, observations = model.simulate(
                theta=model.true_parameters(), final_time=horizon - 1, seed=rep_seed
            )
        # observations is [T, 2]; take first horizon steps
        all_obs.append(observations.numpy()[:horizon])

    return np.array(all_obs), np.array(replication_seeds)


def generate_ksc(model_seed, horizon, replications):
    """Generate from KSC-SV model using canonical frozen-dataset generator."""
    from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
        generate_frozen_exact_sv_dataset_tf,
    )
    from bayesfilter.testing.ksc_ukf_neutra_target_tf import (
        transformed_ksc_observations,
    )

    all_obs = []
    replication_seeds = []

    for rep in range(replications):
        rep_seed = model_seed + 10000 + rep
        replication_seeds.append(rep_seed)

        # Generate raw SV trajectory, then apply KSC transform
        with tf.device("/CPU:0"):
            _states, raw = generate_frozen_exact_sv_dataset_tf(
                seed=rep_seed, horizon=horizon
            )
            observations = tf.cast(transformed_ksc_observations(raw), DTYPE)
            if observations.shape.rank == 1:
                observations = observations[:, None]

        all_obs.append(observations.numpy()[:horizon])

    return np.array(all_obs), np.array(replication_seeds)


def generate_gen_sv(model_seed, horizon, replications):
    """Generate from generalized SV model."""
    from bayesfilter.highdim.ledh_canonical_models_tf import generalized_sv_canonical_model

    theta0_np = [float(np.arctanh(0.7)), float(np.arctanh(0.6)), -0.4, -0.6, 0.1]
    theta0 = tf.constant(theta0_np, DTYPE)
    model, _ = generalized_sv_canonical_model(theta0)

    rho_s, rho_h = np.tanh(theta0_np[0]), np.tanh(theta0_np[1])
    sigma_s, sigma_h = np.exp(theta0_np[2]), np.exp(theta0_np[3])
    beta = np.exp(theta0_np[4])

    rng = np.random.default_rng(model_seed)
    all_obs = []
    replication_seeds = []

    for rep in range(replications):
        rep_seed = model_seed + 10000 + rep
        replication_seeds.append(rep_seed)
        rep_rng = np.random.default_rng(rep_seed)

        s = rep_rng.normal(0.0, sigma_s / np.sqrt(1.0 - rho_s**2))
        hh = rep_rng.normal(0.0, sigma_h / np.sqrt(1.0 - rho_h**2))
        obs_traj = []

        for t in range(horizon):
            s = rho_s * s + sigma_s * rep_rng.standard_normal()
            hh = rho_h * hh + sigma_h * rep_rng.standard_normal()
            y = beta * s + np.exp(0.5 * hh) * rep_rng.standard_normal()
            obs_traj.append([y])

        all_obs.append(np.array(obs_traj))

    return np.array(all_obs), np.array(replication_seeds)


def generate_austria(model_seed, horizon, replications):
    """Generate from Austria SIR model using its canonical .simulate() method."""
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    model = zhao_cui_sir_austria_model()

    all_obs = []
    replication_seeds = []

    for rep in range(replications):
        rep_seed = model_seed + 10000 + rep
        replication_seeds.append(rep_seed)

        # Call the model's canonical .simulate() — RK4 with zhao_cui_sir_step variant
        with tf.device("/CPU:0"):
            states, all_observations = model.simulate(
                final_time=horizon, seed=rep_seed
            )
        # all_observations is [T+1, 9]; frozen target uses y1:y20 (observations[1:21])
        observations = all_observations[1 : horizon + 1]
        all_obs.append(observations.numpy())

    return np.array(all_obs), np.array(replication_seeds)


MODEL_REGISTRY = {
    "linear2d": generate_linear2d,
    "dlgssm": generate_dlgssm,
    "predator-prey": generate_predator_prey,
    "KSC": generate_ksc,
    "gen-SV": generate_gen_sv,
    "Austria": generate_austria,
}


def main():
    parser = argparse.ArgumentParser(description="Generate R2 tuning data")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--seed", type=int, required=True, help="Model seed (9001-9006)")
    parser.add_argument("--horizon", type=int, required=True, help="Trajectory length")
    parser.add_argument("--replications", type=int, required=True, help="Number of trajectories")
    parser.add_argument("--output-dir", default="docs/benchmarks/r2_tuning_20260827")

    args = parser.parse_args()

    print(f"Generating tuning data: model={args.model}, seed={args.seed}, T={args.horizon}, R={args.replications}")

    # Generate
    generator = MODEL_REGISTRY[args.model]
    observations, replication_seeds = generator(args.seed, args.horizon, args.replications)

    # Compute SHA-256
    obs_bytes = observations.tobytes()
    sha256 = hashlib.sha256(obs_bytes).hexdigest()

    # Save
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, f"{args.model}_tuning_data.npz")

    np.savez(
        outpath,
        observations=observations,
        replication_seeds=replication_seeds,
        model_seed=args.seed,
        horizon=args.horizon,
        replications=args.replications,
        sha256=sha256,
    )

    print(f"Saved: {outpath}")
    print(f"  Shape: {observations.shape}")
    print(f"  SHA-256: {sha256[:16]}...")
    print(f"  Replication seeds: {replication_seeds[0]} ... {replication_seeds[-1]}")


if __name__ == "__main__":
    main()
