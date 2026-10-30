"""CPU-only synthetic checks for the source analysis and its real dispatcher call chain."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import fcntl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_controlled_smc_kdm as runner
spec = importlib.util.spec_from_file_location("source_analysis", ROOT / "docs/benchmarks/analyze_controlled_smc_author_results.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
import numpy as np


def source_indices():
    indices = {}
    for key, (_, grids, horizon, particles, iterations, _) in analysis.DRIVERS.items():
        indices[key] = {}
        for grid in range(1, grids + 1):
            for rep in range(1, 101):
                # Synthetic deterministic variation, with deliberately different variances.
                value = -100 - grid + (rep - 50.5) * (.02 if key.endswith("csmc") else .08)
                indices[key][grid, rep] = dict(grid_index=grid, repetition=rep,
                    seconds=.01, cpu_seconds=.009, logZ=value, finite_real=True,
                    N=particles or 6144, I=iterations, T=horizon, ESS=None, refinements=4 if key.startswith("lorenz") else iterations)
    return indices


def make_fixture(campaign):
    """Use the locked source identity with clearly SYNTHETIC numerical observations."""
    locks = list(runner.CAMPAIGN.glob("attempt*_c0_source_lock/source-lock.json"))
    if not locks:
        raise unittest.SkipTest("integration check needs the local C0 source snapshot")
    lock = locks[-1]
    output = campaign / "attempt01_c1_source_full"
    output.mkdir(parents=True)
    indices = source_indices()
    drivers = []
    for key, (lane, grids, horizon, particles, iterations, seed) in analysis.DRIVERS.items():
        dest = output / "reference-work" / lane
        dest.mkdir(parents=True, exist_ok=True)
        author = lock.parent / "author-source/StateSpaceModels" / lane
        for name in (analysis.FILES[key], "cSMC_Resample.m", 'SimulatedData.mat' if lane=='Lorenz' else 'Data_NeuroScience.mat'): 
            shutil.copyfile(author / name, dest / name)
        instrumented = dest / (Path(analysis.FILES[key]).stem + "_full.m")
        instrumented.write_text("% SYNTHETIC TEST DRIVER; never executed\n")
        rows = list(indices[key].values())
        from scipy.io import savemat
        variable=('cSMC' if key.endswith('csmc') else 'APF' if key.endswith('apf') else 'BPF')+'_LogNormConst'
        native={variable:np.array([[indices[key][g,r]['logZ'] for g in range(1,grids+1)] for r in range(1,101)])}
        native.update(N=particles or 6144,I=iterations,T=horizon,nrep=100,ngrid=grids)
        savemat(dest/(key+'-result.mat'),native)
        with (dest / (key + "-cases.csv")).open("w") as stream:
            writer = csv.writer(stream)
            for row in rows:
                writer.writerow([row[k] for k in ("grid_index","repetition","seconds","logZ")] +
                    [1, row["N"],row["I"],row["T"], "nan",row["refinements"],row["cpu_seconds"]])
        drivers.append(dict(driver=key,lane=lane,mode="full",seed=seed,returncode=0,
            status="COMPLETE",cases=rows,expected_cases=grids*100,exact_case_coverage=True,
            source_numerical_failure=False,source_driver_sha256=analysis.sha(dest / analysis.FILES[key]),
            source_algorithm_sha256=analysis.sha(dest / "cSMC_Resample.m"),
            instrumented_sha256=analysis.sha(instrumented),
            source_settings=dict(N=particles or 6144,I=iterations,T=horizon,nrep=100,ngrid=grids),
            native_matrix_validation=dict(shape=[100,grids],checked_cases=grids*100,csv_values_match=True)))
    summary = dict(status="PASS",scope="AUTHOR_SOURCE_MATRIX",author_algorithm_modified=False,
        source_lock=str(lock.parent),source_lock_sha256=analysis.sha(lock),drivers=drivers,
        runtime_equivalence="SYNTHETIC_TEST_ONLY")
    (output / "summary.json").write_text(json.dumps(summary,allow_nan=False))
    manifest = dict(phase="c1_source_full",status="COMPLETE",decision={"status":"PASS"},
        author_commit=analysis.AUTHOR_COMMIT,cpu_only=True,resource="cpu",charged_seconds=0,
        output_root=str(output),output_sha256={str(p.relative_to(output)):analysis.sha(p)
        for p in output.rglob("*") if p.is_file()})
    (output / "manifest.json").write_text(json.dumps(manifest))
    return output, summary, manifest


class StatisticsChecks(unittest.TestCase):
    def test_sample_variance_and_normalization(self):
        values = [-100 + k / 10 for k in range(100)]
        stats = analysis.descriptive(values)
        self.assertAlmostEqual(stats["sample_variance_logZ"], statistics.variance(values), places=12)
        shifted = analysis.descriptive([v + analysis.LORENZ_SHIFT for v in values])
        self.assertAlmostEqual(shifted["sample_variance_logZ"], stats["sample_variance_logZ"], places=10)
        self.assertNotEqual(shifted["relative_variance"],stats["relative_variance"])

    def test_paired_scale_invariance_and_independent_variation(self):
        values = np.arange(1,101,dtype=float)
        paired, draws = analysis.bootstrap_cell([values,values*3],paired=True,cell_seed=2)
        self.assertTrue(paired["valid"])
        np.testing.assert_allclose(draws,0,atol=3e-15)
        independent, other = analysis.bootstrap_cell([values,values*3],paired=False,cell_seed=2)
        self.assertTrue(independent["valid"])
        self.assertGreater(float(np.std(other)),.01)

    def test_zero_variance_veto(self):
        result, logs = analysis.bootstrap_cell(np.ones((2,100)),paired=True,cell_seed=2)
        self.assertFalse(result["valid"])
        self.assertIsNone(logs)

    def test_known_simultaneous_interval_and_ranking(self):
        cells = [dict(comparison=dict(relative_variance_ratio=.25)) for _ in range(2)]
        draws = [math.log(.25)+np.array([-.2,0,.2])]*2
        result = analysis.simultaneous(cells,draws)
        self.assertEqual(result["verdict"],"SUPPORTED_LOWER_RV_THROUGHOUT_GRID")
        self.assertAlmostEqual(result["simultaneous_log_halfwidth"],.2)
        np.testing.assert_allclose(cells[0]["comparison"]["simultaneous_95_interval"],
                                   [.25*math.exp(-.2),.25*math.exp(.2)])
        self.assertEqual(analysis.simultaneous(cells,[None,None])["verdict"],"VETO")

    def test_source_failure_is_not_dropped(self):
        indices = source_indices()
        indices["lorenz_csmc"][1,1].update(finite_real=False,logZ=None)
        results = analysis.analyze_indices(indices)
        self.assertEqual(results[0]["cells"][0]["failures"],[1,0])
        self.assertEqual(results[0]["cells"][0]["csmc"]["n"],99)
        self.assertTrue(results[0]["cells"][0]["csmc"]["conditional_on_finite_only"])
        self.assertEqual(results[0]["grid_inference"]["verdict"],"VETO")
        self.assertEqual(results[1]["grid_inference"]["verdict"],"VETO")
        self.assertEqual(results[2]["grid_inference"]["verdict"],"SUPPORTED_LOWER_RV_THROUGHOUT_GRID")


class SourceAndCallChainChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="c2-synthetic-check-")
        cls.campaign = Path(cls.temp.name)
        cls.source, cls.summary, cls.manifest = make_fixture(cls.campaign)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_fixture_loads(self):
        _,indices,_=analysis.load_source(self.source)
        self.assertEqual(sum(map(len,indices.values())),12000)

    def test_missing_duplicate_and_wrong_settings_rejected(self):
        for change in ("missing","duplicate","settings"):
            driver=copy.deepcopy(self.summary["drivers"][0])
            if change=="missing": driver["cases"].pop()
            if change=="duplicate": driver["cases"][-1]=driver["cases"][0]
            if change=="settings": driver["source_settings"]["N"]=500
            with self.subTest(change=change),self.assertRaises(ValueError):
                analysis.validate_driver(driver["driver"],driver)

    def test_incomplete_attempt_rejected(self):
        path=self.source/"manifest.json"
        original=path.read_bytes()
        try:
            manifest=copy.deepcopy(self.manifest)
            manifest["status"]="RUNNING"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"not complete"): analysis.load_source(self.source)
        finally:
            path.write_bytes(original)

    def test_altered_csv_rejected(self):
        path=self.source/"reference-work/Lorenz/lorenz_csmc-cases.csv"
        original=path.read_bytes()
        try:
            path.write_bytes(original+b"bad\n")
            with self.assertRaisesRegex(ValueError,"hash mismatch"): analysis.load_source(self.source)
        finally:
            path.write_bytes(original)

    def test_native_matrix_disagreement_rejected_even_with_valid_hash(self):
        from scipy.io import loadmat, savemat
        path=self.source/'reference-work/Lorenz/lorenz_csmc-result.mat'
        manifest_path=self.source/'manifest.json'
        original,manifest_original=path.read_bytes(),manifest_path.read_bytes()
        try:
            data={k:v for k,v in loadmat(path).items() if not k.startswith('__')}
            data['cSMC_LogNormConst'][0,0]+=1
            savemat(path,data)
            manifest=json.loads(manifest_original)
            manifest['output_sha256'][str(path.relative_to(self.source))]=analysis.sha(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'native values differ'):
                analysis.load_source(self.source)
        finally:
            path.write_bytes(original)
            manifest_path.write_bytes(manifest_original)

    def test_real_dispatcher_invokes_analyzer_with_synthetic_source(self):
        (self.campaign/"opening-budget.json").write_text(json.dumps({"attempt_limit":4}))
        def synthetic_status():
            records=runner.campaign_attempts()
            return dict(attempts=len(records),running=[str(p) for p,m in records if m["status"]=="RUNNING"],
                passed=[m["phase"] for _,m in records if m["status"]=="COMPLETE"],
                remaining_seconds={"cpu":1000,"gpu":0},tranche_remaining_seconds={"cpu":1000,"gpu":0},
                additional_resource_grants_seconds={"cpu":0,"gpu":0})
        with patch.object(runner,"CAMPAIGN",self.campaign), patch.object(runner,"status",synthetic_status), \
             patch.object(runner,"refresh_checkpoint"):
            result=runner.execute("c2_source_results")
        self.assertEqual(result["status"],"COMPLETE")
        manifest=json.loads((Path(result["output_root"])/"manifest.json").read_text())
        self.assertEqual(Path(manifest["command"][2]).name,"analyze_controlled_smc_author_results.py")
        self.assertIn("--source-attempt",manifest["command"])
        self.assertNotIn("--fixture",manifest["command"])
        self.assertEqual(manifest["seeds"],[analysis.SEED])
        self.assertEqual(manifest["environment"]["CUDA_VISIBLE_DEVICES"],"-1")
        self.assertEqual(manifest["decision"]["replication"],"INCOMPLETE")
        self.assertEqual(manifest["plan"],runner.C2_SOURCE_RESULTS_PLAN)
        report=json.loads((Path(result["output_root"])/"summary.json").read_text())
        self.assertEqual(len(report["experiments"]),3)
        self.assertEqual(report["experiments"][0]["cells"][0]["refinement_counts"],[{"4":100},{"4":100}])
        self.assertEqual(report["source_runtime_equivalence"],"SYNTHETIC_TEST_ONLY")


class SourceFailureReportingChecks(unittest.TestCase):
    def test_full_analysis_rejects_known_source_failure_before_queueing_or_waiting(self):
        current=dict(passed=[],running=['source'],source_driver_failures=[
            dict(driver='lorenz_csmc',status='SOURCE_OR_RUNTIME_FAILURE')])
        with patch.object(runner,'status',return_value=current):
            with self.assertRaisesRegex(RuntimeError,'blocked by incomplete or failed source drivers'):
                runner.continuation_readiness()
        with self.assertRaisesRegex(RuntimeError,'blocked by incomplete or failed source drivers'):
            runner.queue_wait_state('c2_source_results',current)

    def test_partial_source_failure_is_visible_while_other_drivers_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            (output/'progress.json').write_text(json.dumps([
                dict(driver='lorenz_csmc',status='SOURCE_OR_RUNTIME_FAILURE',returncode=1,cases=[{}]),
                dict(driver='lorenz_apf',status='BLOCKED_CSMC_ITERATIONS'),
                dict(driver='neuroscience_csmc',status='COMPLETE',returncode=0,cases=[{}])]))
            records=[(None,dict(phase='c1_source_full',output_root=tmp,status='RUNNING'))]
            with patch.object(runner,'campaign_attempts',return_value=records):
                failures=runner.source_driver_failures()
            self.assertEqual([x['driver'] for x in failures],['lorenz_csmc','lorenz_apf'])
            self.assertEqual(failures[0]['completed_cases'],1)

    def test_failure_replay_waits_for_unrelated_live_source_work(self):
        record=dict(phase='c1_source_full',status='RUNNING',started_epoch=time.time(),
            timeout_seconds=259200,supervisor_pid=123,supervisor_identity='test')
        current=dict(passed=['c0_source_lock','c1_source_isolation_check'],running=['source'])
        with patch.object(runner,'campaign_attempts',return_value=[(None,record)]),patch.object(runner.shared,'alive',return_value=True):
            ready,deadline=runner.queue_wait_state('c1_source_failure_replay',current)
        self.assertFalse(ready)
        self.assertGreater(deadline,time.time()+259100)


class SourceRetryBudgetChecks(unittest.TestCase):
    def check(self,remaining,passed,source_charge=5438):
        current=dict(remaining_seconds={'cpu':remaining},tranche_remaining_seconds={'cpu':remaining},passed=passed)
        records=[(None,dict(phase='c1_source_full',status='FAILED',charged_seconds=source_charge))]
        with patch.object(runner,'campaign_attempts',return_value=records),patch.object(runner.shared,'charge',return_value=source_charge):
            return runner.full_source_timeout(current)

    def test_retry_reserves_analysis_without_new_compute(self):
        self.assertEqual(self.check(253000,['c1_source_isolation_check']),252400)

    def test_retry_requires_exact_source_check(self):
        with self.assertRaisesRegex(RuntimeError,'exact numerical and RNG'):
            self.check(253000,[])

    def test_retry_rejects_insufficient_unchanged_driver_budget(self):
        with self.assertRaisesRegex(RuntimeError,'unchanged source-driver limits'):
            self.check(248999,['c1_source_isolation_check'])


class QueueChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="c2-queue-check-")
        self.path=Path(self.temp.name)
        self.patcher=patch.object(runner,"CAMPAIGN",self.path)
        self.patcher.start()
        self.current=dict(running=["source"],passed=[],remaining_seconds={"cpu":260000},
                          tranche_remaining_seconds={"cpu":260000})
        self.source=dict(phase="c1_source_full",status="RUNNING",started_epoch=time.time(),
                         timeout_seconds=259200,supervisor_pid=os.getpid(),supervisor_identity="test")

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def test_wait_deadline_covers_72_hours(self):
        with patch.object(runner,"campaign_attempts",return_value=[(self.path,self.source)]), \
             patch.object(runner.shared,"alive",return_value=True):
            ready,deadline=runner.queue_wait_state("c2_source_results",self.current)
        self.assertFalse(ready)
        self.assertGreater(deadline-time.time(),259000)

    def test_failed_and_dead_predecessor_rejected(self):
        for status,alive in (("FAILED",True),("RUNNING",False)):
            source=dict(self.source,status=status)
            with self.subTest(status=status),patch.object(runner,"campaign_attempts",return_value=[(self.path,source)]), \
                 patch.object(runner.shared,"alive",return_value=alive),self.assertRaises(RuntimeError):
                runner.queue_wait_state("c2_source_results",self.current)

    def test_cpu_margin_required_before_queueing(self):
        self.current["remaining_seconds"]["cpu"]=259300
        with patch.object(runner,"status",return_value=self.current), \
             patch.object(runner,"campaign_attempts",return_value=[(self.path,self.source)]), \
             patch.object(runner.shared,"alive",return_value=True),patch.object(runner.shared,"charge",return_value=0), \
             self.assertRaisesRegex(RuntimeError,"margin"):
                runner.continuation_readiness()

    def test_duplicate_start_reuses_existing_supervisor(self):
        with patch.object(runner,"queue_status",return_value=dict(status="WAITING",supervisor_alive=True)), \
             patch.object(runner,"progress",return_value={}),patch.object(runner.subprocess,"Popen") as spawn:
            result=runner.start_detached("c2_source_results")
        self.assertEqual(result["launch"]["status"],"WAITING")
        spawn.assert_not_called()

    def test_direct_duplicate_cannot_overwrite_queue(self):
        path=self.path/"queue-c2_source_results.lock"
        with path.open("a+") as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError,"live queue"):
                runner.wait_and_execute("c2_source_results")
        self.assertFalse((self.path/"queue-c2_source_results.json").exists())

    def test_predecessor_failure_writes_terminal_queue_without_launch(self):
        with patch.object(runner,"continuation_readiness"),patch.object(runner,"status",return_value=self.current), \
             patch.object(runner,"refresh_checkpoint"),patch.object(runner.time,"sleep"), \
             patch.object(runner,"queue_wait_state",side_effect=[(False,time.time()+100),RuntimeError("source failed")]), \
             patch.object(runner,"execute") as execute,self.assertRaisesRegex(RuntimeError,"source failed"):
            runner.wait_and_execute("c2_source_results")
        execute.assert_not_called()
        queue=json.loads((self.path/"queue-c2_source_results.json").read_text())
        self.assertEqual(queue["status"],"FAILED")

    def test_wait_serializes_before_execute(self):
        result=dict(phase="c2_source_results",status="COMPLETE",decision={"status":"PASS"})
        with patch.object(runner,"continuation_readiness"),patch.object(runner,"status",return_value=self.current), \
             patch.object(runner,"refresh_checkpoint"), \
             patch.object(runner,"queue_wait_state",side_effect=[(False,time.time()+100),(True,time.time()+100)]), \
             patch.object(runner,"execute",return_value=result) as execute, \
             patch.object(runner.time,"sleep",side_effect=lambda _: execute.assert_not_called()):
            runner.wait_and_execute("c2_source_results")
        execute.assert_called_once_with("c2_source_results")


if __name__=="__main__":
    unittest.main(verbosity=2)
