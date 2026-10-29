"""CPU-only result-assembly guards; no TensorFlow import or GPU work."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from scripts import surrogate_hmc_computation_summary as summary


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class ComputationSummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/"kernel.py").write_text("source")
        (self.root/"snapshot").mkdir()
        (self.root/"snapshot/kernel.py").write_text("source")
        self.ledger=dict(authorized_gpu_wall_seconds=100.,used_seconds=10.,remaining_seconds=90.,max_attempts=120,
                         attempts=[dict(status="FINISHED",wall_seconds=10.)])

    def records(self,name="lgssm"):
        scope=dict(model=name,horizon=20 if name=="range_bearing" else 50,particles=4032,chunk_size=2016,
                   state_dimension=3,parameter_dimension=4,tf32=False,jit_compile=True,force_batch_size=2,
                   dtype="float64" if name=="sir_d18" else "float32",pairwise_dtype="float32",
                   precision_policy="fp64_recursion_fp32_pairs" if name=="sir_d18" else "uniform",
                   source_sha256={"kernel.py":summary.digest(self.root/"kernel.py")})
        selection=dict(status=summary.STATUS["selection"],scope=scope,selected=dict(valid=True,controls={"steps":2},beta=[.3,.7]))
        signature=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()
        selection["selection_hash"]=signature
        records={k:dict(status=v,scope=deepcopy(scope),selection_hash=signature) for k,v in summary.STATUS.items()}
        records["selection"]=selection
        for key in ("paired","precision"):
            records[key].update(controls={"steps":2},beta=[.3,.7])
        records["mechanics"]["pilot"]=dict(finite=True,divergences=0,force_fallback_count=0)
        records["paired"].update(healthy=True,maximum_scaled_force_gap=.002,tolerance=.01,candidate_baseline_warm_ratio=.8)
        records["precision"].update(comparisons={k:dict(finite=True,passed=True,maximum_scaled_error=0.,tolerance=t)
               for k,t in (("likelihood",.01),("score",.01),("force",.01),("binding_wiring",1e-6))},
               health={k:[True]*4 for k in ("candidate","reference","actual_binding")},
               source_snapshot_checks=dict(all_match=True,mismatches=[]),source_snapshot=str(self.root/"snapshot"))
        records["cost"].update(healthy=True,long_chain_cost_model={"classification":"descriptive"},measurements=[
            dict(name=k,warmed_seconds=[1.,2.],cold_compile_execute_seconds=3.,allocator={})
            for k in ("actual_force_B4_in_B2_groups","independent_endpoint_B4")])
        return records

    def fixture(self):
        c=self.root/"campaign"; a=c/"advance"; b=c/"paired"; p=c/"precision"
        write(c/"budget.json",self.ledger)
        write(a/"advance.json",{"status":"COMPUTATION_CHECKS_COMPLETE_POSTERIOR_NOT_RUN"})
        write(b/"next-phase-plan.json",{"status":"PAIRED_CONSUMER_CHECKS_COMPLETE"})
        write(p/"next-phase-plan.json",{"status":"SELECTED_CONSUMER_PRECISION_COMPLETE"})
        for name in summary.MODELS:
            r=self.records(name)
            paths=dict(selection=a/name/"filter-tuning"/name/"filter-selection.json",mechanics=a/name/"mechanics"/name/"result.json",
                       cost=a/name/"consumer-cost"/name/"consumer-cost.json",paired=b/name/name/"consumer-paired.json",
                       precision=p/name/"selected-consumer-precision.json")
            for key,path in paths.items(): write(path,r[key])
        return c,a,b,p

    def test_valid_record_and_complete_result(self):
        self.assertEqual(summary.validate_model("lgssm",self.records(),self.root),[])
        c,a,b,p=self.fixture(); result=summary.assemble(c,a,b,p,self.root)
        self.assertEqual(result["status"],summary.COMPLETE)
        self.assertFalse(result["posterior_sampling_complete"])
        self.assertFalse(result["canonical_admission"])
        self.assertIn("1%",summary.render(result))

    def test_missing_model_blocks_completion(self):
        c,a,b,p=self.fixture(); (p/"sir_d18/selected-consumer-precision.json").unlink()
        result=summary.assemble(c,a,b,p,self.root)
        self.assertEqual(result["status"],summary.REPAIR)
        self.assertTrue(next(x for x in result["models"] if x["model"]=="sir_d18")["issues"])

    def test_failed_selection_without_candidate(self):
        r=self.records(); r["selection"].update(status="FAILED",selected=None)
        self.assertTrue(summary.validate_model("lgssm",r,self.root))

    def test_scope_selection_and_controls_mismatch(self):
        for key,field,value in (("precision","selection_hash","wrong"),("paired","controls",{}),("cost","scope",{})):
            with self.subTest(key=key):
                r=self.records(); r[key][field]=value
                self.assertTrue(summary.validate_model("lgssm",r,self.root))

    def test_campaign_dtype_is_checked_even_if_hash_is_valid(self):
        r=self.records(); scope=r["selection"]["scope"]; scope["dtype"]="float64"
        selection=r["selection"]; selection.pop("selection_hash")
        signature=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()
        selection["selection_hash"]=signature
        for key in ("mechanics","cost","paired","precision"):
            r[key].update(scope=deepcopy(scope),selection_hash=signature)
        self.assertIn("campaign scope mismatch",summary.validate_model("lgssm",r,self.root))

    def test_stale_live_source_and_snapshot_are_checked(self):
        for path in (self.root/"kernel.py",self.root/"snapshot/kernel.py"):
            r=self.records(); path.write_text("changed")
            self.assertTrue(summary.validate_model("lgssm",r,self.root))
            path.write_text("source")

    def test_relaxed_tolerance_nonfinite_or_false_health_blocks(self):
        for field,value in (("tolerance",.02),("maximum_scaled_error",float("nan")),("passed",False)):
            r=self.records(); r["precision"]["comparisons"]["score"][field]=value
            self.assertTrue(summary.validate_model("lgssm",r,self.root))
        r=self.records(); r["precision"]["health"]["actual_binding"][2]=False
        self.assertTrue(summary.validate_model("lgssm",r,self.root))

    def test_budget_validation(self):
        self.assertEqual(summary.ledger_issues(self.ledger),[])
        for field,value in (("wall_seconds",-1.),("wall_seconds",float("nan")),("status","RUNNING")):
            r=deepcopy(self.ledger); r["attempts"][0][field]=value
            self.assertTrue(summary.ledger_issues(r))
        r=deepcopy(self.ledger); r["remaining_seconds"]=None
        self.assertTrue(summary.ledger_issues(r))

    def test_nonterminal_parent_blocks(self):
        c,a,b,p=self.fixture(); write(b/"next-phase-plan.json",{"status":"RUNNING"})
        self.assertEqual(summary.assemble(c,a,b,p,self.root)["status"],summary.REPAIR)

    def test_active_worker_preserves_root_pointer(self):
        c,a,b,p=self.fixture(); self.ledger["attempts"][0]["status"]="RUNNING"; write(c/"budget.json",self.ledger)
        pointer=c/"next-phase-plan.json"; write(pointer,{"keep":"running"})
        args=SimpleNamespace(campaign_ledger=c/"budget.json",advance_dir=a,paired_dir=b,selected_precision_dir=p,output=c/"out")
        with patch("builtins.print"):
            self.assertEqual(summary.finalize(args,self.root,write),1)
        self.assertEqual(summary.read(pointer),{"keep":"running"})
        self.assertFalse(summary.read(c/"out/result.json")["posterior_sampling_complete"])

    def test_success_refreshes_pointer_and_preserves_existing_output(self):
        c,a,b,p=self.fixture()
        args=SimpleNamespace(campaign_ledger=c/"budget.json",advance_dir=a,paired_dir=b,selected_precision_dir=p,output=c/"out")
        with patch("builtins.print"):
            self.assertEqual(summary.finalize(args,self.root,write),0)
        old=(c/"out/result.json").read_bytes()
        self.assertEqual(summary.read(c/"next-phase-plan.json")["status"],summary.COMPLETE)
        with self.assertRaises(FileExistsError): summary.finalize(args,self.root,write)
        self.assertEqual((c/"out/result.json").read_bytes(),old)

if __name__=="__main__": unittest.main()
