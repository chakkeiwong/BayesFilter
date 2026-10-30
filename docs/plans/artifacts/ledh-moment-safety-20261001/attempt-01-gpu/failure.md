# Attempt 01: initialization failure

No numerical comparison executed. The driver imported project modules before
configuring GPU growth. An imported module initialized TensorFlow, making the
subsequent memory-policy configuration fail closed. Repair: configure and verify
growth immediately after importing TensorFlow, before importing project modules.
Trusted standalone GPU probe passed; this is harness ordering, not device failure.
Complete log: command.log. Next attempt uses a fresh directory.
