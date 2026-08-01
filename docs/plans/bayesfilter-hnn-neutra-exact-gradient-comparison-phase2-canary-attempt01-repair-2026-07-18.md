# Phase 2 PP-UKF Canary Attempt 01 Repair Record

Classification: `HARNESS_IMPORT_ORDER_FAILURE`.

The first PP-UKF canary initialized the TensorFlow logical GPU while importing
the comparison module, before the runner called the repository memory-growth
policy. The policy correctly failed closed with `Physical devices cannot be
modified after being initialized`.

This attempt produced no HNN, exact-gradient, target, accuracy, or performance
evidence. It used no serious sampling budget beyond process startup.

Repair: configure TensorFlow GPU memory growth immediately after importing
TensorFlow/TFP and before importing the comparison module. Add a source-order
regression. Re-run focused CPU-hidden tests, then retry PP-UKF in fresh
`attempt-02`. Target, chart, methods, criteria, hardware, and budget remain
unchanged.
