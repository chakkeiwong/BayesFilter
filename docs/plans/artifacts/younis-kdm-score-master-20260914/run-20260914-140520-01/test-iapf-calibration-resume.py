"""CPU-only driver regression: fake endpoints, no numerical research evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("calibration_driver",HERE/"run-iapf-fit-calibration.py")
driver=importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.row={"id":"fixture","proposal":"iapf","iapf":{"max_iterations":4}}

    def test_prior_launch_is_charged_and_actual_fits_release_reservation(self):
        budget=driver.AttemptBudget(32,64,lambda:None)
        wrapped=budget.wrap(lambda row,ctx:{"diagnostics":{"work_accounting":{"offline_fit_calls":2}}},"fixture")
        wrapped(self.row,{})
        self.assertEqual(budget.used,35)
        self.assertEqual(budget.charges[0]["recursive_fits"],2)

    def test_refusal_precedes_numerics_and_does_not_spend(self):
        budget=driver.AttemptBudget(61,64,lambda:None)
        called=[]
        with self.assertRaises(driver.BudgetStop):
            budget.wrap(lambda *args:called.append(True),"fixture")(self.row,{})
        self.assertFalse(called)
        self.assertEqual(budget.used,61)
        self.assertFalse(budget.deferred[0]["numerical_work_started"])

    def test_unknown_failure_charges_upper_bound(self):
        budget=driver.AttemptBudget(32,64,lambda:None)
        def fail(*args):raise RuntimeError("interrupted fit")
        with self.assertRaises(RuntimeError):budget.wrap(fail,"fixture")(self.row,{})
        self.assertEqual(budget.used,36)

    def test_missing_fit_accounting_fails_closed(self):
        budget=driver.AttemptBudget(32,64,lambda:None)
        with self.assertRaises(ValueError):
            budget.wrap(lambda *args:{"diagnostics":{}},"fixture")(self.row,{})
        self.assertEqual(budget.used,36)

    def test_last_comparator_uses_one_charge(self):
        budget=driver.AttemptBudget(63,64,lambda:None)
        budget.wrap(lambda *args:{"diagnostics":{}},"fixture")({"id":"ekf","proposal":"ekf"},{})
        self.assertEqual(budget.used,64)

    def test_budget_interrupt_is_preserved_by_real_coordinator(self):
        from bayesfilter.score_study.coordinator import execute
        from bayesfilter.score_study.registry import default_registry
        old=HERE/"iapf-fit-calibration-gpu-01/weak-calibration-study.json"
        study=json.loads(old.read_text())
        study["settings"].update(device="CPU",tf32=False)
        budget=driver.AttemptBudget(64,64,lambda:None)
        with tempfile.TemporaryDirectory(prefix="iapf-driver-regression-") as tmp:
            output=Path(tmp)/"run"
            with self.assertRaises(driver.BudgetStop):
                execute(study,default_registry(),output,
                    endpoint_loader=lambda path:budget.wrap(lambda *args:self.fail("numerical work started"),"fixture"))
            state=json.loads((output/"state.json").read_text())
            self.assertEqual(state["execution_status"],"incomplete")
            self.assertEqual(state["attempts"][0]["status"],"interrupted")
            self.assertEqual(budget.used,64)

    def test_preserved_weak_results_and_selection_link_are_current(self):
        from bayesfilter.score_study.coordinator import fingerprint
        from bayesfilter.score_study.contracts import digest,validate_result
        from bayesfilter.score_study.registry import default_registry
        registry=default_registry()
        previous=HERE/"iapf-fit-calibration-gpu-01"
        for name in ("weak-calibration","weak-evaluation"):
            state=json.loads((previous/name/"state.json").read_text())
            study=state["study"]
            self.assertEqual(state["fingerprint"],fingerprint(study,registry))
            for row in study["rows"]:
                saved=state["rows"][row["id"]]
                result=json.loads((previous/name/saved["result_path"]).read_text())
                self.assertEqual(digest(result),saved["result_digest"])
                validate_result(result,row,registry)
                if row["role"]=="claim":
                    # Numerical selection re-derivation uses the original GPU
                    # settings and is checked in the trusted resumed launch.
                    selection=json.loads(Path(row["tuning_selection"]).read_text())
                    self.assertEqual(Path(selection["source_run"]),previous/"weak-calibration")


if __name__=="__main__":unittest.main(verbosity=2)
