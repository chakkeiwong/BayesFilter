# Phase 32 Repair and Refresh Note

| Attempt | Failure class | Repair | Result |
|---|---|---|---|
| seed 2802 | none | unchanged theta-measure pilot protocol | `PASS_THETA_MEASURE_PILOT` (`phase32-replication/seed2802/`) |
| seed 2803 | none | unchanged theta-measure pilot protocol | `PASS_THETA_MEASURE_PILOT` (`phase32-replication/seed2803/`) |
| aggregate | none | size/seed-aware descriptive aggregation; no pooled authority metric | `PASS_THETA_REPLICATION_HARD_GATES_DESCRIPTIVE_UNCERTAINTY` (`phase32-replication/aggregate/`) |

The aggregate must preserve each seed root and report hard vetoes separately
from descriptive mean/SD/MCSE. No pooled metric may be promoted to authority
or posterior evidence. The earlier `pending` labels were stale documentation
and are repaired here; the receipts themselves were not rewritten.
