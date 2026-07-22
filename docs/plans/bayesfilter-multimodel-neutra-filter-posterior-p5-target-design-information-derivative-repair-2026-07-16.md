# P5 Information-Derivative Harness Repair

Date: 2026-07-16

The focused post-attempt-02 regression showed that replacing explicit history
`TensorArray` objects was insufficient: TensorFlow reverse-mode
`batch_jacobian` itself creates an XLA-CPU-incompatible variant TensorList when
differentiating through the UKF `tf.while_loop`.

The target-design information diagnostic now obtains derivatives from a single
batched central-FD graph. For B design points it evaluates all `B * 2 * 5`
coordinate perturbations together, with no Python loop over time, parameters,
or proposals. Fine step `5e-5` supplies the information matrix; coarse step
`1e-4` is an independent stability check. Both predictive-mean and log-
innovation-variance derivatives must have maximum scale-normalized step gap at
most `5e-3`.

This repair affects only the prospective likelihood-information diagnostic.
The structural likelihood's production value/score remains the manual analytic
principal-square-root path and retains its independent centered-FD regression.
The target, data, information formula, source points, horizons, eigenvalue/rank/
condition gates, seeds, hardware class, and budget are unchanged. No scientific
row was produced before this repair.
