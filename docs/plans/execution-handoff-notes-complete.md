# Handoff Notes Complete — Ready for Execution

**Date:** 2026-09-07  
**Authority:** ledh-surrogate-hmc-executable-master-program-2026-09-07.md

---

## Summary

Four execution handoff notes written:

1. **Phase 0:** Implement three Option A decisions (tolerance, seed policy, coverage)
   - Time: 1 day
   - GPU: No
   - Deliverables: 3 files + tests

2. **Phase 1:** Re-run existing diagnostics (JVP parity, Sinkhorn, GPU memory growth)
   - Time: 0.5 day
   - GPU: Yes (escalated, diagnostics only)
   - Deliverables: Test results + GPU verification

3. **Phase 2:** Toy potential test (3D quadratic, exact vs damped force)
   - Time: 0.5 day
   - GPU: Yes (0.5 GPU-hour)
   - Deliverables: W₂ agreement result

4. **Phase 3:** Seed policy verification (V1 determinism, V2 reversibility, V3 no-call-count)
   - Time: 1 day
   - GPU: No
   - Deliverables: Dual-adapter + 3 tests

---

## Total Handoff Budget

**CPU time:** 3 days (Phases 0-3)  
**GPU time:** 0.5 GPU-hour (Phase 2 only)

**After Phase 3 completes:**
- Phase 4a: 2 GPU-hours (diagnostic)
- Phase 4b: 6-9 GPU-hours (certification, conditional)
- Phase 5: 24-33 GPU-hours (3 models, conditional)

---

## Key Features of Handoff Notes

### 1. **Bounded I/O Instructions**
Each handoff specifies:
- Use `grep` to find code, don't read full files
- Read only targeted line ranges (offset/limit)
- Write outputs to files, read only summaries
- No full pytest output in terminal

### 2. **Clear Success Criteria**
Each phase has:
- Primary criterion (pass/fail)
- Veto diagnostics (blocking)
- Explanatory diagnostics (informational)
- Result summary JSON format

### 3. **Failure Handling**
Each phase specifies:
- What to do on failure
- Diagnosis file to write
- Whether failure blocks next phase
- No proceeding without user decision

### 4. **Completion Notes**
Each phase writes:
- `docs/plans/phaseN-complete.md`
- `results/phaseN-summary.json`
- Status, artifacts, next phase

---

## Execution Flow

### **Start with Phase 0** (this session, clean context)
- Implement tolerance derivation
- Write seed policy memo
- Implement joint Mahalanobis coverage
- Run smoke tests
- Write Phase 0 completion note

### **Then Phase 1** (same session if context allows, or new session)
- Read Phase 0 completion note (10 lines)
- Re-run existing diagnostics
- Verify GPU memory growth
- Write Phase 1 completion note

### **Then Phase 2** (new session recommended, GPU work)
- Read Phase 1 completion note
- Run toy potential test
- Measure W₂ distance
- Write Phase 2 completion note

### **Then Phase 3** (new session)
- Read Phase 2 completion note
- Build dual-adapter
- Run V1, V2, V3 tests
- Write Phase 3 completion note

### **Then Phase 4a** (new session, 2 GPU-hours)
- Read Phase 3 completion note
- Read Phase 4a handoff note (to be written)
- Run ultra-short LGSSM diagnostic
- Decision gate: pass → Phase 4b, fail → STOP

---

## Handoff Note Effectiveness

### **What they prevent:**
1. ✅ Context thrashing (bounded I/O, no full-file reads)
2. ✅ Ambiguous success criteria (explicit pass/fail)
3. ✅ Proceeding after failure (explicit stop instructions)
4. ✅ Lost state across sessions (completion notes, result JSONs)

### **What they enable:**
1. ✅ New agent can start any phase from completion note only
2. ✅ Phases are independently executable (no cross-phase context)
3. ✅ Result summaries are small (< 1KB JSON each)
4. ✅ User can inspect progress via completion notes

---

## Files Written

```
docs/plans/
├── phase0-execution-handoff.md  (3.2 KB)
├── phase1-execution-handoff.md  (2.8 KB)
├── phase2-execution-handoff.md  (4.1 KB)
└── phase3-execution-handoff.md  (3.9 KB)
```

---

## Next Action

**Start Phase 0 execution immediately.**

**Command to begin:**
```
Execute Phase 0 per docs/plans/phase0-execution-handoff.md
```

Agent will:
1. Create `bayesfilter/inference/tolerance_derivation.py`
2. Write `docs/memos/ledh-surrogate-hmc-seed-policy-2026-09-07.md`
3. Create `bayesfilter/inference/coverage.py`
4. Write tests
5. Run tests (CPU-only)
6. Write `docs/plans/phase0-complete.md`
7. Write `results/phase0-summary.json`

**Estimated time:** 1 day  
**Context usage:** Low (3 new files + 2 test files, no large reads)

---

## User Decision Required

**Shall I start Phase 0 execution now?**

If yes, I will:
- Work from the Phase 0 handoff note
- Follow bounded I/O instructions
- Write completion note when done
- Report status

If no:
- Ready to answer questions about handoff notes
- Ready to revise any handoff note before execution
- Ready to wait for your signal
