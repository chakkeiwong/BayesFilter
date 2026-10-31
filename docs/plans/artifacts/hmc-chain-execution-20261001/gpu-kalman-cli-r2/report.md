# HMC chain execution benchmark

Hardware diagnostic only. Fastest observed is descriptive, not a tuning or convergence decision.

| Layout | Batch / workers / cores per worker | Startup s | First HMC s | Warm HMC s | Transitions/s | Host peak sum MiB |
|---|---|---:|---:|---:|---:|---:|
| 0 | 2 / 1 / 2 | 4.057 | 6.429 | 0.176307 | 68.063 | 1200.0 |
| 1 | 1 / 2 / 1 | 3.997 | 6.504 | 0.037593 | 319.212 | 2392.0 |

Timings include synchronization, IPC and tensor artifacts. Host peak sums are sums of worker
lifetime high-water marks, not simultaneous live memory. GPU allocator peaks are reset per call.
Target timings, allocator memory, parity, health, failures and provenance are in result.json.
An eligible short timing run does not establish convergence, posterior validity or tuning authority.
