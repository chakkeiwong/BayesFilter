"""Failure and interpretation guards for the exploratory sensitivity report."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


def module(name):
    path=ROOT/'docs/benchmarks'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


EXPERIMENT=module('diagnose_lorenz_random_perturbation')
REPORT=module('summarize_lorenz_perturbation')


class SensitivityReportingTests(unittest.TestCase):
    def example(self):
        variants=EXPERIMENT.variants()
        def row(v,delta,solver=None):
            result=dict(case=0,forcing=2.5,seed=20260929,variant=v['id'],valid=True,
                zero_control_exact=v['id']=='zero',terminal_logZ=-1000.+delta,
                stages=[dict(stage=0,ancestry_changed=0)],stage_count=1,error='')
            if solver:result['solver']=solver
            return result
        source=[row(v,0 if v['id']=='zero' else 100 if v['id']=='normal_d0_s-1' else 1) for v in variants]
        port=[row(v,5,'example_solver') for v in variants]
        return source,port,variants

    def test_outlier_does_not_become_tolerance(self):
        result=REPORT.summarize(*self.example())
        arm=next(a for a in result['cases'][0]['arms'] if a['arm']=='normal')
        self.assertEqual(arm['source_abs_delta_logZ']['maximum'],100)
        self.assertEqual(arm['source_abs_delta_logZ']['median'],1)
        self.assertFalse(arm['baseline_difference_within_source_median'])
        self.assertIsNone(result['current_port_acceptable'])

    def test_failure_veto_survives_small_successful_errors(self):
        source,port,variants=self.example()
        port[1].update(valid=False,terminal_logZ=None,error='INVALID_FITTED_POLICY')
        result=REPORT.summarize(source,port,variants)
        self.assertEqual(result['additional_tf_failures'],1)
        self.assertFalse(result['current_port_acceptable'])

    def test_antithetic_design_and_likelihood_units(self):
        variants=EXPERIMENT.variants()
        self.assertEqual(len(variants),29)
        for arm in ('normal','uniform','combined'):
            for direction in range(4):
                signs=[v['sign'] for v in variants if v['arm']==arm and v['direction']==direction and v['epsilon']==1e-8]
                self.assertEqual(sorted(signs),[-1,1])
        import math
        self.assertAlmostEqual(REPORT.percent(math.log(1.02)),2.)


if __name__=='__main__':unittest.main()
