# Claude Audit Result: HMC Tuning Interface Documentation Plan

Date: 2026-08-27

Reviewer: Claude (Fable 5)

Plan reviewed: `/home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-hmc-tuning-interface-documentation-and-verification-plan-2026-08-27.md`

Baseline commit: `553208502e2e43e6883ad9467381eb5c3e82867a`

Review protocol: READ-ONLY BOUNDED REVIEW per companion handoff memo

## FINDINGS

No material findings. The plan is technically and scientifically sufficient to produce a correct, falsifiable, agent-usable account of BayesFilter's HMC tuning interfaces.

### Minor clarity observations (non-blocking)

**[MINOR] Capability registry architecture is labeled but not yet proven**
Plan anchor: Section 8, row "Capability registry as shared table source"
Status: The plan correctly labels this as `architecture hypothesis` and gates it with registry validation, generation tests, and evidence-anchor requirements in Phase 2 (Section 14, lines 510-518). The phased structure and fail-closed unknowns (Section 10, line 278) adequately guard against premature claims.

**[MINOR] ESS characterization requires careful non-invention discipline**
Plan anchor: Section 13.3, lines 439-449
Status: The plan explicitly forbids introducing a new default ESS number (line 444), requires a stop for any new threshold (lines 445-446), and mandates that prose render tested current status including `known defect` if needed (lines 447-449). This meets the required standard.

## OPEN QUESTIONS OR UNCHECKED CLAIMS

None. The plan explicitly states which baseline facts were inspected (Section 3), which claims require behavioral tests before documentation (Sections 13, 16), and which boundaries require owner decisions (Sections 11.2, 14 Phase 7, Section 17). Every material interface claim is tied to either an inspected baseline anchor or a planned focused test.

## AUDIT COVERAGE

All 15 required audit questions from the companion handoff memo are answered:

1. **Two active artifact-authority tuners**: Yes. Section 3 baseline table row 1 cites `tuning_contract.py:143-155` and `test_hmc_tuning_contract.py:22-30`. Section 13.1 requires registry and authority tests. The plan does not treat registry membership as proof of behavior; Section 4 explicitly states "The registry is authoritative for route role and artifact authority only. It is not, by itself, evidence that mass adaptation, trajectory selection, or fresh verification executed correctly" (lines 82-85).

2. **Correct distinction of tuner/transport-tuner/helper/runner**: Yes. Section 9 (lines 221-243) defines all four terms before use. Section 10 human decision tree (lines 280-292) separates the cases. Section 13.5 line 461 requires that direct low-level execution be classified as mechanics-only.

3. **Fixed-transport tuner not an arbitrary-force escape hatch**: Yes. Section 3 baseline row 6 (lines 63) inspects the prerequisite contract. Section 11.2 rejection condition (line 341) explicitly forbids routing arbitrary force through the fixed-transport tuner. Section 13.4 (lines 451-458) tests that a raw arbitrary-force object cannot satisfy the transport prerequisite.

4. **Neural-force limitation stated plainly**: Yes. Section 3 baseline rows 7-9 (lines 64-66) document fixed mass coordinates, fixed `L`, identity fallback, and tracing rejection. Section 9 defines "Neural-force mechanics" (lines 240-243) as not implying fixed transport and not independently tuning mass or `L`. Section 10 step 3 (lines 286-289) states the limitation directly.

5. **Typed runner binding can prove stage consistency**: Yes. Section 11.1 (lines 296-322) enumerates all required binding fields including force identity, exact endpoint-target identity, coordinate scope, mass-wrapper compatibility, arbitrary fixed `L`/`epsilon` support, telemetry, and source closure. Lines 313-317 require threading through every stage. Section 13.5 line 463 requires a fake-runner call ledger showing which stages invoked it and which parameters it received.

6. **Adequate rejection conditions**: Yes. Section 11.2 (lines 324-342) lists six explicit rejection conditions including missing telemetry, unbindable identity, silent algorithm substitution, and misleading contracts. Line 338 requires stopping and writing a separate API decision record if rejected.

7. **Tests independently verify route/stage/verification/admission**: Yes. Section 13 separates registry tests (13.1), ordinary tuner stage tests (13.2), ESS/admission tests (13.3), fixed-transport tests (13.4), neural-force tests (13.5), and negative mutation tests (13.6). Section 13.2 line 435 explicitly requires that a forced verifier result with `passed=False` cannot produce a final handoff.

8. **Forced verifier failure prevents handoff**: Yes. Section 13.2 lines 434-436 state this as a required test. Section 13.6 line 481 includes "a failed fresh verifier may still emit a final handoff" as a false statement that must be rejected. Section 17 line 654 includes this as a stop rule.

9. **ESS work characterizes current behavior without inventing thresholds**: Yes. Section 13.3 lines 439-449 require characterizing whether ESS is disabled/explanatory/required in current config (line 440), testing forced failure when configured as required (lines 441-443), forbidding a new default number (line 444), and rendering tested current status including `known defect` if not repaired (lines 447-449). Section 3 baseline row 10 (lines 67-68) explicitly states the August 22 audit finding must be reproduced before documenting.

10. **Generated documentation drift or false prose can be detected**: Yes. Section 12.5 tracks generated fragments (lines 400-410). Section 13.1 requires byte-for-byte generation tests (line 425) and that changing a capability makes `--check` fail (line 427). Section 13.6 (lines 472-482) requires negative mutation tests that temporarily inject false claims and prove rejection. Section 15 line 603 includes `--check` in the verification matrix. Section 18 pre-mortem addresses this explicitly (line 678).

11. **Numeric thresholds classified by provenance**: Yes. Section 7.5 (lines 161-168) forbids inventing defaults and requires tracing to code/policy/derivation/literature. Section 8 default-and-assumption audit table (lines 203-217) includes a dedicated row for existing numeric thresholds requiring re-audit (line 213). Section 10 line 361 requires numeric-policy provenance references. Section 18 pre-mortem includes a numeric-policy ledger check (line 675).

12. **Target correctness/convergence/performance/readiness kept outside documentation claim**: Yes. Section 5 evidence contract line 97 states "What will not be concluded: No posterior convergence, target correctness, sampler superiority, performance, GPU readiness, production readiness, or scientific validity claim follows from documentation tests." Section 6 line 115 repeats this. Section 9 line 253 includes forbidden-claims requirement. Section 16 line 645 requires nonclaims in the result note.

13. **Downstream phase respects pins and owner boundaries**: Yes. Section 3 baseline rows 11-12 (lines 68-69) document the downstream state. Section 14 Phase 7 (lines 569-580) requires separate reviewable patches, records BayesFilter schema/commit, and explicitly forbids changing the dsge_hmc lock without owner selection (lines 575-577). Section 17 line 659 includes this as a stop rule.

14. **Commands answer separate questions, valid CPU environment, preserve scope**: Yes. Section 7.7 (lines 180-186) requires deliberate CPU-only with `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import. Section 15 (lines 596-618) lists exact commands, all prefixed with CPU hiding or running in CPU-only contexts. Section 7.8 (lines 188-196) explicitly separates which test answers which question. Section 19 lines 682-685 restates CPU-only scope and forbids GPU runs.

15. **Tracked claim-supporting files separated from ignored debris with zero-untracked gate**: Yes. Section 12.5 (lines 400-410) explicitly separates tracked generated fragments from ignored build products. Section 8 rows 15-16 (lines 215-216) document the policy. Section 15 line 617 includes `git ls-files --others --exclude-standard` as the final check. Section 16 line 647 requires this is empty. Section 18 pre-mortem line 678 includes this check.

## AUDIT NOTES

The plan demonstrates exceptional discipline in several areas:

- **Falsifiability**: Every normative claim is tied to either an inspected baseline anchor with exact line numbers or a planned behavioral test with explicit pass/fail conditions. Section 13.6 negative mutations ensure false claims can be detected.

- **Provenance transparency**: Section 3 measured baseline table, Section 8 default audit, and Section 7.5 numeric-policy rules ensure no inherited value hardens into an unsupported default merely by repetition.

- **Boundary preservation**: Mathematical targets (Section 2 lines 43-49), scientific claims (Section 5 line 97, Section 6 line 115), downstream locks (Section 14 Phase 7), and owner decisions (Section 11.2, Section 17) are all explicitly excluded from implicit authorization.

- **Stop conditions**: Section 7.3, Section 11.2, Section 13.3, Section 14 Phase 4, and Section 17 all contain explicit stop rules preventing documentation of untested behavior or silent policy changes.

- **Source-of-truth order**: Section 4 establishes a clear hierarchy with checked mathematical targets and executable behavior above registry, registry above generated docs, and prose as the final layer. The plan does not invert this.

- **Neural-force design**: Section 11 correctly treats the preferred compatibility design as a hypothesis requiring full-stage conformance tests (Section 11.1) and provides explicit rejection conditions (Section 11.2) rather than assuming feasibility.

- **ESS non-invention**: Section 13.3 is particularly careful, requiring characterization of current behavior (line 440), forbidding new defaults (line 444), requiring a stop for threshold selection (lines 445-446), and accepting `known defect` documentation if repair is out of scope (lines 447-449).

The plan's phased structure (Section 14) sequences work correctly: baseline lock → contradiction inventory → capability registry → interface decision → admission characterization → documentation → build → downstream → closeout. Each phase has a concrete gate. The verification matrix (Section 15) separates registry checks, behavioral tests, examples, and build validation into distinct commands that answer distinct questions.

## VERDICT: AGREE

The plan is technically and scientifically sufficient. It can produce a correct, falsifiable, agent-usable account of BayesFilter's HMC tuning interfaces, including the ordinary, fixed-transport, and neural-force boundaries. Implementation may proceed after status is changed per Section 21 and the companion handoff protocol.
