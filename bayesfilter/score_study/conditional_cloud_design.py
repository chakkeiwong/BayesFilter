"""Frozen design for conditional particle-cloud diagnostics, not admission."""

PARTICLE_COUNTS = (32, 64)
REPLICATES = 12
ARMS = ("identity", "current_global", "future_global", "current_local",
        "current_equal", "current_calibrated_current", "current_calibrated_future",
        "future_calibrated_future")
HEURISTICS = ("identity", "current_global", "current_local", "current_equal")
FIXED_BETAS = {"identity": [1., 0., 0.], "global": [.1, .9, 0.],
               "local": [.1, 0., .9], "equal": [1. / 3.] * 3}


def data_seed(regime, partition, dataset):
    return 317000 + 1000 * regime + 100 * partition + dataset


def cloud_seed(regime, partition, dataset, replicate, particles):
    seed = 417000 + 10000 * regime + 1000 * partition + 100 * dataset + replicate
    return seed + (100000 * particles if partition == 0 else 0)


def evaluation_cases():
    for regime in range(2):
        for partition, count in ((1, 1), (2, REPLICATES)):
            for dataset in range(2):
                for replicate in range(count):
                    yield regime, partition, dataset, replicate


def make_clouds(tf, particles, seed):
    if particles not in PARTICLE_COUNTS:
        raise ValueError("unsupported diagnostic particle count")
    initial = tf.random.stateless_normal([64, 2], [seed, 2], dtype=tf.float64)
    normals = tf.random.stateless_normal([5, 64, 2], [seed, 3], dtype=tf.float64)
    uniforms = tf.random.stateless_uniform([5, 64, 2], [seed, 4], dtype=tf.float64)
    return initial[:particles], normals[:, :particles], uniforms[:, :particles]
