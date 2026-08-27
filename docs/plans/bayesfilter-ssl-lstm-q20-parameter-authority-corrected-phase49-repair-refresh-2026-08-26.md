# Corrected Parameter-Authority Phase 49 Repair and Refresh

Date: 2026-08-26  
Source result: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase49-result-2026-08-26.md`  
Branch: `depth8_does_not_reduce_variability`  
Next version: `v3.2-defensive-proposal-support`

## Interpretation

The v3.1 depth experiment passed every engineering and numerical boundary. It
showed substantial independent-MH movement, but the three-replicate primary
spread condition was not met: negative-mode spread was larger than the frozen
depth-two comparator, while ESS spread also remained unfavorable. The result
does not establish that independent MH is invalid; it leaves proposal-support
overlap as the smallest unresolved mechanism.

## Repair decision

Run a new paired diagnostic with the same q=20 target, theta measure, initial
clouds, resampling schedule, and eight-step depth. Draw candidates from

`r(theta) = (1-rho) q(theta) + rho Normal(theta; center, tau^2 I)`

with the bounded hypotheses `rho=0.50` and `tau=4.0`. The annealing base
remains `pi_beta proportional to q^(1-beta) exp(beta V)`. Because `r` is not
the base density, the exact independent-MH log ratio is

`bridge_q(theta') - bridge_q(theta) + log r(theta) - log r(theta')`.

The runner must evaluate `r` at both current and candidate rows, reject invalid
target/status candidates, and retain the original q values for the tempering
weights. This tests proposal support rather than silently changing the target.

## Required gates and stop rules

1. A CPU-hidden fixture checks the non-symmetric `q`-base/`r`-proposal ratio at
   beta zero and beta one over repeated steps, including finite states and
   nonzero movement.
2. The q=20 runner reproduces Phase 47 initial and identity hashes and loads a
   passing frozen Phase 49 report as the depth-eight comparator.
3. Candidate rows are sampled from the declared `r` mixture and scored by the
   same normalized `r` density; current/candidate q values remain the base
   bridge values.
4. Invalid candidates are never accepted; all retained tensors remain finite
   `[256,4]` rows in `theta_R4`.
5. The report compares the new support arm with the frozen Phase 49 depth-eight
   arm. Acceptance, ESS, mode mass, roots, and spread remain explanatory; no
   threshold is promoted.

An unavailable target/proposal, exact fixture contradiction, three unrepaired
infrastructure failures, or exhausted campaign budget is a continuation veto.
Poor whitening or a failed support arm is a repair trigger, not a direction
veto.

## Next subplan

`docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase50-subplan-2026-08-26.md`

