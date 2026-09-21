"""Dense spectral pullback checks against an independent full-rank formula."""

import numpy as np
import pytest
import tensorflow as tf

from tests.test_filter_repair_quadratic_numerics import inputs, program

D=tf.float64


@pytest.mark.parametrize('dimension',[3,5])
@pytest.mark.parametrize('case',['regular','near_identity','repeated'])
def test_dense_precision_and_spectral_pullback(dimension,case):
    values=inputs('dense',dimension,case)
    kernel=program('dense',dimension,jit=True)
    reference=program('dense',dimension,source='original')

    def make(selected,jit):
        @tf.function(input_signature=selected.input_signature,jit_compile=jit,autograph=False)
        def evaluate(*arguments):
            with tf.GradientTape() as tape:
                tape.watch(arguments)
                result=selected(*arguments)
                # Frobenius squared agrees with sum of squared eigenvalues;
                # its derivative is well-defined at repeated eigenvalues.
                objective=.5*tf.reduce_sum(tf.square(result['raw_eigenvalues']))
            return objective,tape.gradient(objective,arguments,unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return evaluate

    actual=make(kernel,True)(*values)
    # The frozen original uses MatrixSolveLs(fast=False), for which TensorFlow
    # defines no gradient. It remains the primal authority, not a derivative
    # authority. Graph/current parity and the independent formula check the VJP.
    expected=make(program('dense',dimension,jit=False),False)(*values)
    for left,right in zip(tf.nest.flatten(actual),tf.nest.flatten(expected),strict=True):
        np.testing.assert_allclose(left,right,atol=1e-10,rtol=1e-10)
    raw=reference(*values)['raw_precision'].numpy()
    np.testing.assert_allclose(actual[0],.5*np.sum(raw*raw),atol=1e-10,rtol=1e-10)
    center,design,scores,checks,check_scores=(value.numpy() for value in values)
    response=center[None,:]-scores
    coefficient=np.linalg.lstsq(design,response,rcond=None)[0]
    symmetric=.5*(coefficient+coefficient.T)
    adjoint=np.linalg.solve(design.T@design,symmetric)
    response_gradient=design@adjoint
    design_gradient=(response-design@coefficient)@adjoint.T-response_gradient@coefficient.T
    independent=(.5*np.sum(symmetric*symmetric),(
        np.sum(response_gradient,axis=0),design_gradient,-response_gradient,
        np.zeros_like(checks),np.zeros_like(check_scores)))
    for left,right in zip(tf.nest.flatten(actual),tf.nest.flatten(independent),strict=True):
        np.testing.assert_allclose(left,right,atol=1e-10,rtol=1e-10)
