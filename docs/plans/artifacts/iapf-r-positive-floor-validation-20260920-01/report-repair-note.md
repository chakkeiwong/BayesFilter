# Local reporting repair

The first attempt to invoke the completed d80-B reporter found d80-A's tail
diagnostic still running. Its manifest had no final `wall_seconds`, causing
`KeyError: 'wall_seconds'` in the reporting wrapper's budget calculation.
The report process had not started and no result or scientific run changed.
Repair: reserve a running diagnostic's timeout, as the replication launcher
already does, then use its actual time when complete. This is a routine
localized reporting repair within the original post-run budget. Charge one
second conservatively for the failed wrapper invocation. The existing worker
manifests and complete filter evidence remain intact.
