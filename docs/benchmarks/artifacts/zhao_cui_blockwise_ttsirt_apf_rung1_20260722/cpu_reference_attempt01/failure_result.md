# Zhao-Cui Blockwise TTSIRT-APF Rung-1 Failed Attempt

Status: `FAIL_INFRASTRUCTURE_OR_HARNESS`

The 16-candidate fit and TTSIRT branch compilation completed, but the matched
exact-reference harness incorrectly indexed the vector returned by
`tf.searchsorted`, producing scalar ancestors instead of `[N]`. No APF result
was evaluated.

The localized repair removed that obsolete scalar indexing. A CPU shape smoke
then passed for both exact arms with ancestors `[2,8]` and states `[3,8,2]`.

This attempt is not candidate evidence and does not reject the research
direction.
