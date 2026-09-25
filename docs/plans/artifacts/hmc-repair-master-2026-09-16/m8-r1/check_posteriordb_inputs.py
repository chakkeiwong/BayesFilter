"""CPU reference preflight; reference means never supply a tuning decision."""
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0,str(REPO))
from bayesfilter.testing.inference_validation.posteriordb_targets import load_case,PosteriordbTarget,CASES
from bayesfilter.testing.inference_validation.storage import write_json
from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
from bayesfilter.inference.hmc_precision import mean_precision


def main():
    rows = []
    for case in CASES:
        content = load_case(REPO/".localresources/posteriordb-20260918",case)
        target = PosteriordbTarget(case,content["data"],jit_compile=False)
        metadata = content["reference_info"]
        assert all(metadata["checks_made"].values())
        assert not any(metadata["diagnostics"]["divergent_transitions"])
        reference = tf.stack([tf.stack([tf.constant(c[name],tf.float64) for name in target.model_names],axis=-1)
                              for c in content["reference_chains"]],axis=1)
        assert reference.shape == (1000,10,target.parameter_dim)
        rhat = rank_normalized_split_rhat_summary(reference,rhat_max=1.01)
        assert rhat["passed"]
        mean = mean_precision(reference,method="lugsail",jit_compile=False)
        assert bool(tf.reduce_all(mean["valid"]))
        q = np.random.default_rng(264).normal(size=(4,target.parameter_dim))
        # Independent analytic score fixture also accepts these exact supplied data.
        import importlib.util
        spec = importlib.util.spec_from_file_location("reference_test",REPO/"tests/inference_validation/test_posteriordb_targets.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = [module.reference(case,content["data"],point) for point in q]
        value,score = target.log_prob_and_grad(q)
        np.testing.assert_allclose(score,[v[1] for v in expected],rtol=1.e-10,atol=1.e-10)
        np.testing.assert_allclose(value.numpy()-value[0],np.array([v[0] for v in expected])-expected[0][0],rtol=1.e-10,atol=1.e-10)
        rows.append({"case":case,"inputs":content["files"],"target_signature":target.adapter_signature(),
            "source_rhat_max":max(metadata["diagnostics"]["r_hat"]),"recomputed_rhat":rhat,
            "means":mean["estimate"].numpy().tolist(),"mcse":mean["mcse"].numpy().tolist(),
            "model_names":target.model_names,"checked_exact_data_score":True,
            "reference_exact_or_iid_claim":False})
    write_json(ROOT/"posteriordb-input-check.json",{"rows":rows,"passed":True,"command":sys.argv})
    print([(r["case"],r["recomputed_rhat"]["max_finite_rhat"]) for r in rows])


if __name__ == "__main__":
    main()
