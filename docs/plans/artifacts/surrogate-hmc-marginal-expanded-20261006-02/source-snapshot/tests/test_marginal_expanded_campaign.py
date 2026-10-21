"""CPU-only harness diagnostics; not scientific or GPU promotion evidence."""
import importlib.util
from pathlib import Path
import math
import statistics
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import pytest
import tensorflow as tf
from bayesfilter.testing import marginal_weight_campaign as base
from bayesfilter.testing import marginal_expanded as expanded


def test_fresh_datasets_are_independent_and_repeatable():
    spec=base.specification("lg3-t10")
    a=base.dataset(spec,"claim",dataset_id=0)
    b=base.dataset(spec,"claim",dataset_id=1)
    again=base.dataset(spec,"claim",dataset_id=0)
    assert bool(tf.reduce_all(a==again))
    assert not bool(tf.reduce_all(a==b))
    assert not bool(tf.reduce_all(a==base.dataset(spec,"validation",dataset_id=0)))
    assert not bool(tf.reduce_all(a==base.dataset(spec,"claim")))


@pytest.mark.parametrize("d",[1,3,10,20])
def test_lg_dimension_grid_is_stable_and_has_declared_shapes(d):
    spec=base.specification(f"lg{d}-t10",tf.float64)
    assert base.dataset(spec,"claim",dataset_id=0).shape==(10,d)
    for point in spec["points"]:
        mean,cov=spec["model"].initial_distribution_fn(tf.constant(point,tf.float64))
        assert mean.shape==(d,)
        assert bool(tf.reduce_all(tf.linalg.eigvalsh(cov)>0))
        assert bool(tf.reduce_all(tf.math.abs(tf.linalg.eigvalsh(spec["a"]*point[0]))<1))
    if d==3:
        expected=tf.constant([[.6,.1,0],[.1,.6,.1],[0,.1,.6]],tf.float64)
        assert bool(tf.reduce_all(spec["a"]==expected))


def fake_datasets():
    datasets=[]
    for shift in (-.2,-.1,0.,.1):
        rows=[dict(ancestor=dict(value=10.,score=[1.],valid=True),
                   marginal_mixture=dict(value=10.+shift,score=[1.],valid=True)) for _ in range(32)]
        datasets.append(dict(reference=dict(valid=True,rows=[dict(value=0.,score=[0.])],
            uncertainty=[dict(value=0.,score=0.)]),points=[dict(rows=rows)]))
    return datasets


def test_uncertainty_clusters_by_dataset_and_missing_baselines_cannot_pass():
    result=expanded.summary(fake_datasets(),1)[0]
    expected=statistics.stdev([-.2,-.1,0.,.1])/2
    assert math.isclose(result["metrics"]["value"]["standard_error"],expected,rel_tol=1e-12)
    assert result["heuristic_dominance_verdict"]=="INCOMPLETE"


def test_bad_reference_or_incomplete_designs_invalidates_summary():
    data=fake_datasets()
    data[0]["reference"]["valid"]=False
    assert expanded.summary(data,1)[0]["verdict"]=="INVALID_CANDIDATE"
    data=fake_datasets()
    data[0]["points"][0]["rows"].pop()
    assert expanded.summary(data,1)[0]["verdict"]=="INVALID_CANDIDATE"


def test_precision_designs_are_identical_after_cast():
    spec=base.specification("lg3-t10")
    single=base.designs(spec,32,"engineering",0,tf.float32,dataset_id=2)
    double=base.designs(spec,32,"engineering",0,tf.float64,dataset_id=2)
    assert all(bool(tf.reduce_all(tf.cast(a,tf.float64)==b)) for a,b in zip(single,double))


@pytest.mark.parametrize("name",["lg3-t2","ksc-t2","m13-t2"])
def test_every_reference_consumer_traces_and_preserves_legacy_values(name):
    # CPU/XLA reference smoke: fresh observations must use the same graph.
    spec=base.specification(name,tf.float32)
    spec64=base.specification(name,tf.float64)
    kernels=expanded.Kernels(spec)
    helper=expanded.comparator_module()
    seed=tf.constant([731,91],tf.int32)
    points=tf.constant(spec["points"][0],tf.float64)
    kinds=["bootstrap","prior","gaussian"]
    kinds.append("kalman" if name.startswith("lg") else "ksc" if name.startswith("ksc") else "fisher")
    for kind in kinds:
        function=kernels.get(kind,32,18.,4)
        first=None
        for idx in (0,1):
            observed=base.dataset(spec,"engineering",dataset_id=idx)
            if kind=="kalman": legacy=base.kalman_reference(spec64,observed)(points)
            elif kind=="ksc": legacy=base.ksc_grid_reference(spec64,observed,32,18.)(points)
            elif kind=="fisher": legacy=base.m13_fisher_reference(spec64,observed,32,4)(points,seed)
            elif kind=="bootstrap": legacy=dict(value=base.bootstrap_reference(spec64,observed,32)(points,seed))
            elif kind=="prior": legacy=dict(value=helper.prior_trajectory_reference(tf,base,spec64,observed,32)(points,seed))
            else: legacy=helper.gaussian_reference(tf,base,spec64,observed)(points)
            dynamic=function(points,observed,seed)
            assert bool(tf.reduce_all(tf.math.is_finite(dynamic["value"])))
            for field in ("value","score"):
                if field in legacy:
                    tf.debugging.assert_near(dynamic[field],legacy[field],rtol=1e-8,atol=1e-8)
            if first is None: first=dynamic["value"]
            else: assert float(first)!=float(dynamic["value"])
        assert function.experimental_get_tracing_count()==1
        if kind in ("bootstrap","prior","gaussian"):
            fp32=kernels.get(kind,32,dtype=tf.float32)
            result=fp32(tf.cast(points,tf.float32),tf.cast(observed,tf.float32),seed)
            assert result["value"].dtype==tf.float32
            assert bool(tf.reduce_all(tf.math.is_finite(result["value"])))
            if kind=="gaussian":
                tf.debugging.assert_near(tf.cast(result["value"],tf.float64),dynamic["value"],atol=.003,rtol=.001)
