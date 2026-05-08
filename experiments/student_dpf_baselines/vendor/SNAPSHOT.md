# Student Baseline Vendor Snapshot

Snapshot date: 2026-05-08

These repositories are vendored only for internal experimental comparison.
They are not production BayesFilter code.

| Source | Branch | Commit | Commit date | Local path |
| --- | --- | --- | --- | --- |
| `2026MLCOE` | `main` | `020cfd7f2f848afa68432e95e6c6e747d3d2402d` | `2026-02-27T13:47:48-06:00` | `2026MLCOE/` |
| `advanced_particle_filter` | `main` | `d2a797c330e11befacbb736b5c86b8d03eb4a389` | `2026-04-25T20:20:51-05:00` | `advanced_particle_filter/` |

Quarantine rule:

```text
No production module under bayesfilter/ may import from experiments/.
```

Update rule:

Do not track moving upstream branches for baseline reports.  If newer student
code is needed, record a new snapshot event with a new commit hash and result
note before rerunning comparisons.
