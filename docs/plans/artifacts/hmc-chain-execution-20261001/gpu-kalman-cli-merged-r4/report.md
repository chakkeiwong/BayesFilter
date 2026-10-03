# HMC chain execution benchmark

Hardware diagnostic only. Fastest observed is descriptive, not a tuning or convergence decision.

| Layout | Batch / workers / cores per worker | Startup s | First HMC s | Warm HMC s | Transitions/s | Host peak sum MiB |
|---|---|---:|---:|---:|---:|---:|
| 0 | 2 / 1 / 2 | 3.716 | 6.157 | 0.175923 | 68.212 | 1195.9 |
| 1 | 1 / 2 / 1 | 3.905 | 6.149 | 0.152020 | 78.937 | 2392.0 |

Timings include synchronization, IPC and tensor artifacts. Host peak sums are sums of worker
lifetime high-water marks, not simultaneous live memory. GPU allocator peaks are reset per call.
Target timings, allocator memory, parity, health, failures and provenance are in result.json.
An eligible short timing run does not establish convergence, posterior validity or tuning authority.
