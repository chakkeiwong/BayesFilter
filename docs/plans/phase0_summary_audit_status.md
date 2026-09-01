# Phase 0 Summary: Route Identity and Audit Status

**Date:** August 30, 2026  
**Worktree:** `/home/chakwong/BayesFilter/.claude/worktrees/ledh-canonical-rebuild`  
**Commit:** 39f0b20e512e495fa2d42b4c72ce7bcbd4daf01f

---

## Audit Status

**Document:** `docs/requests/ledh_score_discrepancy_audit_response_2026_08_29.md`  
**Status:** **BLOCKED_FOR_CLAIM** (diagnostic result; research direction remains open)

### Key Findings from Audit

The existing score-discrepancy audit identified the following issues:

1. **Registry and route wiring defects** — score/value comparison used different controls
2. **Correction-family mismatch** — different reset routes
3. **Dtype inconsistency** — float32 vs float64 in some paths
4. **Random-stream differences** — seed policies not matched
5. **Diagnostic exposure gaps** — transport residuals, cap activity not logged
6. **Tuning scope** — score lane needs tuning on disjoint data

### Audit Disposition

> "The correct disposition is to preserve the discrepancy as a diagnostic, repair the registry
> and route wiring, match the complete score control family and dtype, expose diagnostics, tune
> the score scope on disjoint data, and then rerun a common-cloud, replicated comparison."

**Translation:** The existing analytical score has wiring issues that prevent it from being
claimed as a valid derivative of the value.

---

## Surrogate-Force Strategy

**Key insight:** Surrogate-force HMC **does not require** value-score parity. It explicitly
uses a *different* configuration for the force than for the value.

### Why Surrogate-Force Can Proceed

1. **Different configs by design:**
   - Value: λ=1e-5, δ=1e-5 (exact)
   - Score: λ=1e-3, δ=1e-3 (damped)
   - No claim that score = ∇value

2. **Self-contained frozen noise:**
   - Dual adapter generates its own noise deterministically from θ
   - Does not rely on existing value-score wiring

3. **Testing mechanics, not accuracy:**
   - Phase 1 tests HMC invariance on toy potential
   - Phase 2 tests mixing and posterior coverage
   - NOT claiming the score is the correct derivative

4. **Bounded claims only:**
   - "Chain samples executed pseudo-posterior deterministically"
   - "Score bias moved out of correctness path"
   - NOT claiming "removes bias" or "exact posterior"

### What Phase 0 Means for This Implementation

**We bypass the audit blockers** because:
- Surrogate-force is a new, standalone implementation
- It does not modify the existing LEDH route
- It does not claim to fix the discrepancy
- Success criteria are mechanical (acceptance, coverage), not parity with Kalman

**Phase 0 tasks reduced to:**
1. ✓ Document audit status (this file)
2. ✓ Verify we're in the correct worktree and commit
3. ✓ Confirm fixture parameters match historical (d=3, T=50, θ, obs seed)
4. Run Phase 1-2 as new code, not as repair of existing route

---

## Route Configuration (For Reference)

Current canonical LEDH configuration used in dual adapter:

```python
reset_epsilon = 1.0
reset_sinkhorn_steps = 8
reset_balance_steps = 8
reset_ridge = 1e-5  # (exact) or 1e-3 (damped)
reset_delta_damping = 1e-5  # (exact) or 1e-3 (damped)
```

Fixture:
```python
d = 3
T = 50
N = 1008
theta_true = [0.72, 0.55, 0.35, 0.35, 0.45]
observation_seed = 81100
```

Exact Kalman reference (for coverage check only):
```python
exact_value = -145.434287
exact_score_phi_1 = -6.216707
```

---

## Gate Decision

**Can Phase 1-2 proceed?** YES

**Rationale:**
- Audit blockers apply to claiming the existing score is correct
- Surrogate-force makes no such claim
- Implementation is standalone, does not modify existing route
- Success criteria are mechanical, not dependent on Kalman parity

**Next action:** Execute Phase 1 (toy potential), then Phase 2 (LGSSM three-arm).

---

## What Cannot Be Claimed

Even if Phase 1-2 pass, we CANNOT claim:

- ❌ "The existing LEDH analytical score is correct"
- ❌ "Score bias is removed"
- ❌ "This is HMC-ready for production DSGE"
- ❌ "The pseudo-posterior is the true posterior"

We CAN claim (if tests pass):

- ✓ "Surrogate-force HMC with damped score has deterministic mechanics"
- ✓ "Acceptance and mixing are acceptable on LGSSM diagnostic"
- ✓ "The chain samples the executed pseudo-posterior, which differs from true by ~0.09% value bias"
- ✓ "Score bias affects mixing, not correctness"

---

## Status: Ready to Execute

Phase 0 complete. Proceeding to Phase 1.
