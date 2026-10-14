"""Source checkpoint transitions for bearings KDM stages, without optimizer carryover.

Source anchors: pinned Younis kde_particle_filter.py:280-355. Partial weight
pretraining transfers three modules; MDPF -> A-MDPF copies missing resampling
modules. These are parameter transfers, not resumable optimizer checkpoints.
"""
import tensorflow as tf

WEIGHT_PREFIXES = ("weight_model.", "observation_encoder.", "particle_encoder_for_weights_model.")


def transfer_stage_parameters(destination, source_state, stage):
    if source_state["dtype"] != destination.dtype.name or source_state["variant"] != "mdpf":
        raise ValueError("Incompatible source stage identity")
    source = source_state["parameters"]
    if stage == "weights_to_mdpf":
        if destination.variant != "mdpf":
            raise ValueError("Weight pretraining must feed MDPF")
        selected = {k:v for k,v in source.items() if k.startswith(WEIGHT_PREFIXES)}
        expected = {k for k in destination.parameters if k.startswith(WEIGHT_PREFIXES)}
        if set(selected) != expected or set(source_state["trainable_names"]) != expected:
            raise ValueError("Incomplete weight-pretraining modules")
    elif stage == "mdpf_to_am_dpf":
        if destination.variant != "am_dpf":
            raise ValueError("Full MDPF stage must feed A-MDPF")
        selected = dict(source)
        for original, target in (("weight_model.", "resampling_weight_model."),
                                  ("weighted_bandwidth_predictor.", "resampling_weighted_bandwidth_predictor.")):
            if any(k.startswith(target) for k in destination.parameters) and not any(k.startswith(target) for k in source):
                selected.update({target+k[len(original):]:v for k,v in source.items() if k.startswith(original)})
        if set(selected) != set(destination.parameters):
            raise ValueError("Incomplete full source stage checkpoint")
    else:
        raise ValueError("Unknown source stage transition")
    assignments = []
    for name, value in selected.items():
        tensor = tf.convert_to_tensor(value, destination.dtype)
        variable = destination.parameters[name]
        if tensor.shape != variable.shape or not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
            raise ValueError("Invalid transferred parameter")
        assignments.append((variable, tensor))
    # Validate complete transfer before mutation. Each source stage creates Adam anew.
    for variable, value in assignments:
        variable.assign(value)
    for variable in destination.moments1 + destination.moments2:
        variable.assign(tf.zeros_like(variable))
    destination.step.assign(destination.c(0))
    return dict(stage=stage, transferred_names=sorted(selected), optimizer_reset=True)
