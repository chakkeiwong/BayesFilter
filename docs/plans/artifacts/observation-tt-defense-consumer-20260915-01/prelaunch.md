# A05 prelaunch

Protocol review AGREE is saved in review-01.txt. Skeptical audit is in A05.
Focused checks: 17 passed initially; the mixed endpoint test failed because the
CLI intentionally defers TensorFlow/lib imports until GPU policy configuration.
The test now installs those module references explicitly, preserving that policy;
all four consumer tests pass on retry. No numerical implementation changed for
this repair. The other 14 tests already passed. Both logs are preserved.

Trusted nvidia-smi identifies RTX 5080 by the wrapper's UUID. Another workload
uses some GPU memory; this bounded diagnostic makes no speed superiority claim.
Memory growth is required and recorded. The numerical attempt has a 300-second
external timeout, a 290-second internal limit, unique output, and no refitting.
All research target rows are generated after safety selection is frozen.
