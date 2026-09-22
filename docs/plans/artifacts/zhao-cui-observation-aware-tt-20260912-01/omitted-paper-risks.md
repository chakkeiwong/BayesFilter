# Omitted-paper and implementation risks

- The bounded pass did not exhaustively survey sigma-point particle filters, auxiliary particle filters, or transport-map filtering papers. A UKF default decision would be premature.
- The local likelihood-weighted chart uses `2d` points without a central point and freezes the guide at a reference parameter. Both choices are extensions and require target-specific tests.
- A coordinate-dependent row distribution changes the regression measure. Reusing fixed Christoffel rows or weights after changing the chart is unsupported until the transformed measure is derived.
- The author MATLAB source uses covariance regularization in `computeL.m`. That ridge is a numerics-altering choice and cannot silently become a local default; it needs a calibration/non-harm check.
- The current local preparation is eager/offline reference code. End-to-end TensorFlow graph compatibility, rank behavior, and paper-scale performance remain unchecked.
