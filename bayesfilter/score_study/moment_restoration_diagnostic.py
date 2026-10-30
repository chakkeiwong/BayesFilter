"""Independent failure localization only; not a training or selection route."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from bayesfilter.score_study.moment_training_campaign import Campaign, tf, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.monotonic()
    campaign = Campaign(argparse.Namespace(output=arguments.output, budget_seconds=600., smoke=False))
    from bayesfilter.highdim import higher_moment_contract_e as correction

    original = json.loads((arguments.source / "manifest.json").read_text())
    source_map = json.loads((arguments.source / "source-file-map.json").read_text())
    checks = {}
    for name, expected in original["source_hashes"].items():
        path = Path(source_map[name]) if name in source_map else Path(name)
        checks[name] = hashlib.sha256(path.read_bytes()).hexdigest() == expected
    allowed_changes = {"ledh_unified_correction_tf.py"}
    if any(not match and Path(name).name not in allowed_changes for name, match in checks.items()):
        raise ValueError("Original captured sources do not match")
    failed = json.loads((arguments.source / "test.jsonl").read_text().splitlines()[-1])
    data = campaign.dataset(failed["dataset_seed"])
    saved = json.loads((arguments.source / f"dataset-{failed['dataset_seed']}.json").read_text())
    matches = {name: bool(tf.reduce_all(data[name] == tf.constant(saved[name], tf.float64)))
               for name in ("observations", "label", "score_label")}
    if not all(matches.values()):
        raise ValueError("Diagnostic input mismatch")
    states, covariances, noises, d_states, d_covariances = campaign.particles_for(failed["particle_seed"])
    value, score, trace = campaign.kernels["csmc"](campaign.theta,
        tf.one_hot(0, 3, dtype=tf.float64), states, covariances, noises,
        data["observations"], d_states, d_covariances,
        *(tf.zeros_like(coefficient) for coefficient in data["csmc"]))
    save(arguments.output / "trace.json", trace)
    signature = [tf.TensorSpec([64, 18], tf.float64), tf.TensorSpec([64], tf.float64),
        tf.TensorSpec([64, 18, 1], tf.float64), tf.TensorSpec([64, 1], tf.float64),
        tf.TensorSpec([64, 18], tf.float64), tf.TensorSpec([64, 18, 1], tf.float64)]

    @tf.function(input_signature=signature, jit_compile=True)
    def inspect_restore(source, weights, source_tangent, weights_tangent, points, points_tangent):
        target = correction._weighted_moments_jvp(source, weights, source_tangent, weights_tangent)
        current = correction._uniform_moments_jvp(points, points_tangent)
        result = {}
        for name, moments in (("target", target), ("current", current)):
            mean, covariance, mean_tangent, covariance_tangent = moments
            safe_cov, safe_tangent = correction._relative_psd_covariance(covariance, covariance_tangent)
            factor = tf.linalg.cholesky(safe_cov)
            factor_tangent = correction._cholesky_jvp(factor, safe_tangent)
            result.update({name + "_mean": mean, name + "_covariance": covariance,
                name + "_mean_tangent": mean_tangent, name + "_covariance_tangent": covariance_tangent,
                name + "_safe_covariance": safe_cov, name + "_safe_tangent": safe_tangent,
                name + "_factor": factor, name + "_factor_tangent": factor_tangent})
        centered = points - result["current_mean"][None, :]
        centered_tangent = points_tangent - result["current_mean_tangent"][None, :, :]
        result["standardized"] = correction._right_solve(result["current_factor"], centered)
        result["standardized_tangent"] = correction._right_solve_jvp(result["current_factor"],
            result["current_factor_tangent"], centered, centered_tangent)
        result.update(correction.affine_restore_cloud_jvp(source, weights, source_tangent,
            weights_tangent, points, points_tangent))
        return result

    findings = []
    for index, step in enumerate(trace[:4]):
        inputs = (step["children"], step["posterior_weights"], step["d_children"][..., None],
            step["d_posterior_weights"][..., None], step["higher_moment_stage_post_cap"],
            step["higher_moment_stage_post_cap_tangent"])
        save(arguments.output / f"restore-input-{index + 1}.json", inputs)
        inspected = inspect_restore(*inputs)
        if "higher_moment_stage_post_cap_offsets" in step:
            offsets = step["higher_moment_stage_post_cap_offsets"]
            offset_tangent = step["higher_moment_stage_post_cap_offsets_tangent"]
            offset_moments = correction._uniform_moments_jvp(offsets, offset_tangent)
            inspected["offset_covariance"] = offset_moments[1]
            inspected["offset_covariance_tangent"] = offset_moments[3]
        save(arguments.output / f"restore-intermediates-{index + 1}.json", inspected)
        summary = {}
        for name, tensor in inspected.items():
            if tensor.dtype.is_floating:
                finite = bool(tf.reduce_all(tf.math.is_finite(tensor)))
                summary[name] = dict(finite=finite, maximum_absolute=float(tf.reduce_max(tf.abs(tensor))))
                if name.endswith("covariance") and finite:
                    eigenvalues = tf.linalg.eigvalsh(tensor)
                    summary[name].update(min_eigenvalue=float(eigenvalues[0]),
                        max_eigenvalue=float(eigenvalues[-1]), trace=float(tf.linalg.trace(tensor)))
        weights = step["posterior_weights"]
        findings.append(dict(time=index + 1, program_valid=step["program_valid"],
            positive_weights=tf.reduce_sum(tf.cast(weights > 0, tf.int32)),
            maximum_weight=tf.reduce_max(weights), moments=summary))
    manifest = dict(campaign.manifest, plan="docs/plans/ledh-moment-restoration-repair-20261020.md",
        result=str(arguments.output / "result.json"), role="numerical root-cause diagnosis only",
        original_source_checks=checks, input_matches=matches,
        inspected_source_hash=hashlib.sha256(Path(correction.__file__).read_bytes()).hexdigest(),
        wall_seconds=time.monotonic() - started, calls=1, findings=findings)
    save(arguments.output / "result.json", manifest)
    save(arguments.output / "manifest.json", manifest)
    print(json.dumps(dict(output=str(arguments.output), wall_seconds=manifest["wall_seconds"],
        step_valid=[bool(item["program_valid"]) for item in findings])))


if __name__ == "__main__":
    main()
