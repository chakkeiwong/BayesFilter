# Phase M2 plan: DPF monograph mathematical architecture and chapter-map design

## Date

2026-05-09

## Purpose

Design the final chapter architecture for the DPF monograph block based on the
Phase M1 literature audit rather than on the previous three-chapter draft.

## Main question

How many chapters are actually needed to treat the topic rigorously and
self-containedly, and how should they be partitioned so that each chapter has a
clear mathematical role?

## Candidate content buckets to resolve

At minimum evaluate whether separate chapters are needed for:

1. particle-filter foundations and likelihood estimators;
2. particle-flow mathematics (EDH, LEDH, PF-PF, stochastic/kernel variants);
3. differentiable resampling and OT resampling;
4. learned / amortized transport operators;
5. implementation-oriented mathematical constraints;
6. HMC target correctness and DSGE/MacroFinance suitability.

## Required output

Produce a chapter map note that defines:

- final chapter count and chapter titles;
- ordering within `docs/main.tex`;
- the mathematical role of each chapter;
- the prerequisite graph among the chapters;
- which chapters are primarily theory, which are implementation-oriented, and
  which deliver the HMC assessment;
- which existing draft chapters should be rewritten, split, merged, or removed.

## Required structure for the note

### A. chapter list

| Proposed chapter | Purpose | Main sources | Main equations / objects |
| --- | --- | --- | --- |

### B. dependency graph

| Chapter | Depends on | Enables |
| --- | --- | --- |

### C. replacement map

| Existing draft artifact | Keep / split / rewrite / retire | Reason |
| --- | --- | --- |

## Hard requirements

- Prefer mathematical coherence over minimizing chapter count.
- Do not force the topic into only three chapters if the literature survey shows
  that more are needed.
- The architecture must be understandable to a monograph reader without repo
  background.

## Exit gate

Proceed only once the target chapter architecture is explicit and justified by
Phase M1 evidence.
