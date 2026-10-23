"""Frozen design for an optional guided-mixture mechanism diagnostic."""

POLICIES = ("global", "local", "equal", "calibrated_current", "calibrated_future")
GUIDANCE = ("current", "future")
ARMS = ("identity",) + tuple(
    f"{guide}_{policy}" for policy in POLICIES for guide in GUIDANCE)
HEURISTICS = ("identity", "current_global", "current_local", "current_equal")
SEED_BASE = 217000


def make_factorial_arms(fixed, calibrated, messages):
    probabilities = {name: fixed[name] for name in ("global", "local", "equal")}
    probabilities.update(calibrated_current=calibrated["current"],
                         calibrated_future=calibrated["exact"])
    arms = [("identity", fixed["identity"], messages["current"])]
    for policy, beta in probabilities.items():
        arms.append((f"current_{policy}", beta, messages["current"]))
        arms.append((f"future_{policy}", beta, messages["exact"]))
    return arms
