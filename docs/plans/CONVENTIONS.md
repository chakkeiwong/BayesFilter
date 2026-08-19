# docs/plans Conventions

Date: 2026-08-20
Status: `ACTIVE_CONVENTION`

Scope: NEW documents added to `docs/plans/` from this date forward. The
existing corpus is append-only evidence and is NOT retro-formatted; do not
rename, rewrite, or reflow historical files to satisfy this note.

Machine consumer: `docs/plans/generate_plans_index.py`, which builds
[INDEX.md](INDEX.md) and [INDEX-FULL.md](INDEX-FULL.md) from filenames,
`Date:`/`Status:` lines, and supersession/correction banners. Following these
conventions is what makes a document findable and its staleness visible in
the index.

## 1. Filename

```
bayesfilter-<lineage>-<role>-YYYY-MM-DD.md
```

- Lowercase kebab-case throughout. Do not use the underscore dialect
  (`bayesfilter_..._2026_06_19.md`) for new files; it exists only in
  historical documents.
- `<lineage>` is the campaign/topic stem shared by every document in the
  same investigation (e.g. `austria-genut-xla-nan-localization`). Reuse the
  existing stem exactly when continuing a lineage — the index groups by it.
- `<role>` is one of the standard suffixes, chosen from (most common first):
  `plan`, `result`, `reset-memo`, `subplan`, `checkpoint`, `handoff`,
  `ledger`, `note`, `spec`, `review-request`, `review-reply`, `runbook`,
  `master-program`. Round/attempt qualifiers go between role and date:
  `...-review-round-02-2026-08-20.md`.
- The date is the creation date and is never updated. Content updates to a
  living document (e.g. an execution checkpoint) keep the original filename;
  the `Date:` line and dated sections inside record the update history.

## 2. Required header block

The first lines of every new document:

```markdown
# <Human Title>

Date: YYYY-MM-DD
Status: `SHORT_MACHINE_READABLE_STATUS`
```

- `Date:` — creation date, `YYYY-MM-DD`, on its own line.
- `Status:` — one backticked SCREAMING_SNAKE token on ONE line. Keep prose
  out of it; elaboration goes in the body. (Multi-line status values
  truncate in the index.) Examples in current use:
  `AUTHORIZED_BOUNDED_FEASIBILITY_TRIAL`, `EXECUTED_DIAGNOSTIC_COMPLETE`,
  `HANDOFF_OPEN_WORK_ITEMS_NO_EXECUTION_STARTED`, `READ_ONLY_AUDIT_COMPLETE`.
- Update the `Status:` line in place when the document's state changes
  (e.g. plan `REVIEWED_FOR_EXECUTION` → `EXECUTED_SEE_RESULT`); preserve
  superseded statuses in the body if the history matters, as the Austria
  checkpoint does ("Superseded former statuses (preserved)").

Only 2,349 of the 6,204 historical files carry a parseable `Status:` line;
new files must not add to the unparseable majority.

## 3. Supersession and correction (append-only discipline)

Never delete or rewrite historical text. Two banner forms, both parsed by
the index into flags:

**Supersession banner** — added at the TOP of the *superseded* document
(directly under the `Date:`/`Status:` header), quoting style:

```markdown
> **Superseded YYYY-MM-DD** by
> `docs/plans/<exact-superseding-path>.md`
> for <which claims are superseded>. <What, if anything, remains valid.>
```

- Name the superseding document by exact path.
- Scope the banner: say *which claims* are superseded. "Partially
  Superseded" is the correct label when only a premise or a status is
  overtaken (see the value-surrogate strategy banner for the pattern).
- Chains are acceptable: if A is superseded by B and B by C, A's existing
  banner pointing to B need not be rewritten — but do not rely on readers
  walking more than one hop; add a direct banner when the drift is material.

**Correction banner** — for a claim now known wrong, at the claim's
location (not only at the top):

```markdown
> **Correction YYYY-MM-DD:** <the claim> is wrong relative to that claim /
> stale. <Evidence pointer.> See `docs/plans/<correcting-doc>.md`.
```

Plain-language discipline applies inside banners: `stale`, `wrong relative
to that claim`, `unsupported`, `not checked` — no softening.

**Obligation:** whoever writes a terminal result or reset memo that
materially supersedes earlier documents must add the banners to those
earlier documents in the same session. This is the step that was
historically skipped (see the 2026-08-19 cleanup,
`bayesfilter-docs-redteam-ledger-2026-08-19.md`); the templates now carry
it as a checklist item.

## 4. Large data does not go in prose files

Do not inline bulk JSON/tensor dumps into a plan/result document (the
historical DPF row-localization results, 10k–20k lines each, are the
anti-pattern; they dominate this directory's 764 MB). Put the data under
`docs/benchmarks/artifacts/` or `docs/plans/artifacts/` and reference it by
path plus SHA-256. A result document quotes only the rows it interprets.

## 5. Index regeneration

After adding or bannering documents:

```bash
python docs/plans/generate_plans_index.py
```

Cheap (~2 s), deterministic, safe to run anytime; it writes only
`INDEX.md` and `INDEX-FULL.md`. Never hand-edit those two files.
