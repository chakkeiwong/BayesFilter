# LEDH/main integration review

The owner requested preserving the completed LEDH campaign, merging it into
main, incorporating origin/main, pushing main and synchronizing the development
branch. Local main first merged sqmc-development at 9b12f6b6f; the remote merge
adds five existing HMC/NeuTra/documentation commits.

Skeptical audit before validation: numerical LEDH sources have no overlapping
remote changes. The only common files are .gitignore, docs/main.tex and
references.bib. The ignore rules and chapter list combine cleanly. The
bibliography conflict consists of appended entries; retain complete entries
from both parents and preserve the two locally corrected metadata entries,
verified against publisher DOI records. All 220 citation keys are retained,
without duplicates. New remote NeuTra policy applies to a separate study.

Validation question: does the merged source retain executable LEDH/reporting
contracts and compile both documents with resolved references? Run the prior
36-check harness plus four reporting tests on CPU with GPUs explicitly hidden;
this is an integration regression, not new research or promotion evidence.
Build both PDFs in a fresh directory, preserve commands/logs, and inspect the
new result table and combined chapter list. A test/build failure or missing
citation is a repair trigger before push. Do not rerun the research campaign
or change frozen settings. Budget: one initial focused check plus one retry
for a localized integration repair; no GPU campaign, tuning or environment
mutation. Existing compile warnings are reported separately.

The campaign findings and reference limitations remain those of the committed
master summary. Git publication here synchronizes the owner-authorized repo;
it does not promote the method to a new numerical default.
