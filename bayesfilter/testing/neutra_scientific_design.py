"""Frozen scientific design; importable without TensorFlow/device initialization."""
from bayesfilter.testing.neutra_target_specifications import fixed_specification, random_mixture_specification

METHODS = ("fab", "gabrie", "ais", "smc", "aft", "craft")
FIT_SEEDS = (51, 52, 53)
CALIBRATION_TARGETS = ("calibration_random_two", "calibration_random_three")
FINAL_TARGETS = ("fixed_unwarped", "fixed_warped", "random_two_1103", "random_two_1104",
                 "random_three_2103", "random_three_2104")
TEACHER_REPLICATIONS = 4
REFERENCE_ROWS = 4096

# Owner direction, 2026-10-06. Scoped to this scientific training campaign;
# historical source snapshots and generic artifact constructors are unchanged.
DEFAULT_STUDENT_KIND = "naf_dsf"
FORWARD_REVERSE_POLICY = "bayesfilter_neutra_naf_forward_reverse_v1"


def forward_reverse_student():
    return {"kind": DEFAULT_STUDENT_KIND, "width": 64, "batch": 64,
            "forward_updates": [8192, 8192], "forward_rates": [0.001, 0.0003],
            "rkl_rates": [0.0001, 0.0003], "rkl_rungs": [256, 1024, 4096],
            "reference_rows": 32768, "gradient_clip": 1000.0, "jit_compile": True}


def forward_reverse_teachers():
    return tuple({"profile_id": f"native_{i}", "particles": n, "stages": stages,
                  "mutation_steps": moves, "step_size": dt,
                  "mode_starts": starts, "mode_iterations": iterations,
                  "replications": 4, "jit_compile": True}
                 for i,n,stages,moves,dt,starts,iterations in (
                     (0,4096,32,8,.05,64,200), (1,8192,64,16,.025,128,400)))


def student_candidates():
    return (
        {"profile_id": "student_small", "width": 32, "updates": 4096,
         "learning_rate": 1e-3, "rkl_updates": 256, "rkl_learning_rate": 3e-4},
        {"profile_id": "student_deep", "width": 64, "updates": 8192,
         "learning_rate": 1e-3, "rkl_updates": 256, "rkl_learning_rate": 3e-4},
    )


def combine_profile(native, student):
    return {**native, "student": dict(student)}


def target_catalog():
    """Return the frozen development/final target catalogue."""
    return {
        "calibration_random_two": random_mixture_specification(2, 9101),
        "calibration_random_three": random_mixture_specification(3, 9201),
        "fixed_unwarped": fixed_specification("mixture"),
        "fixed_warped": fixed_specification("warped_mixture"),
        "random_two_1103": random_mixture_specification(2, 1103),
        "random_two_1104": random_mixture_specification(2, 1104),
        "random_three_2103": random_mixture_specification(3, 2103),
        "random_three_2104": random_mixture_specification(3, 2104),
    }


def profile_candidates(method):
    if method not in METHODS:
        raise ValueError("unknown native method")
    """Two finite candidates per method; values are calibration hypotheses."""
    if method in ("ais", "smc"):
        return (
            {"profile_id": f"{method}_small", "particles": 256, "stages": 8,
             "mutation_steps": 4, "proposal_scale": 4.0,
             "step_size": 0.1, "learning_rate": 1e-3},
            {"profile_id": f"{method}_deep", "particles": 512, "stages": 16,
             "mutation_steps": 8, "proposal_scale": 8.0,
             "step_size": 0.05, "learning_rate": 3e-4},
        )
    if method == "gabrie":
        return (
            {"profile_id": "gabrie_small", "walkers": 16, "walker_steps": 4,
             "proposal_scale": 4.0, "step_size": 0.1,
             "learning_rate": 1e-3},
            {"profile_id": "gabrie_deep", "walkers": 32, "walker_steps": 8,
             "proposal_scale": 8.0, "step_size": 0.05,
             "learning_rate": 3e-4},
        )
    if method == "aft":
        return (
            {"profile_id": "aft_small", "particles": 256, "stages": 8,
             "inner_updates": 2, "passes": 2, "mutation_steps": 4,
             "proposal_scale": 4.0, "step_size": 0.1,
             "learning_rate": 1e-3},
            {"profile_id": "aft_deep", "particles": 512, "stages": 16,
             "inner_updates": 4, "passes": 2, "mutation_steps": 8,
             "proposal_scale": 8.0, "step_size": 0.05,
             "learning_rate": 3e-4},
        )
    if method == "craft":
        return (
            {"profile_id": "craft_small", "particles": 256, "stages": 8,
             "inner_updates": 2, "passes": 2, "mutation_steps": 4,
             "proposal_scale": 4.0, "step_size": 0.1,
             "learning_rate": 1e-3},
            {"profile_id": "craft_deep", "particles": 512, "stages": 16,
             "inner_updates": 4, "passes": 4, "mutation_steps": 8,
             "proposal_scale": 8.0, "step_size": 0.05,
             "learning_rate": 3e-4},
        )
    # FAB's nonlinear alpha-two tail is screened before any native run.
    return (
        {"profile_id": f"{method}_screen_small", "particles": 64,
         "stages": 8, "mutation_steps": 3, "proposal_scale": 4.0,
         "step_size": 0.1, "learning_rate": 1e-3},
        {"profile_id": f"{method}_screen_deep", "particles": 128,
         "stages": 16, "mutation_steps": 4, "proposal_scale": 8.0,
         "step_size": 0.05, "learning_rate": 3e-4},
    )
