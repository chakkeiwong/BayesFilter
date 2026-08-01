# Phase 7 Serious Attempt-2 Pre-Runtime Verification Commands

Date: 2026-07-13

Status: `TERMINAL_NO_RUNTIME_COMMAND_EVIDENCE`

Working directory for every command:
`/home/chakwong/BayesFilter`.

No command in this record built authority, consumed a claim, reserved an
attempt-2 output, created a worker, or executed an HMC/XLA transition.

## Final Decision-Bearing Reconstruction

Exit: `0`. Wall time: `9.0 s`.

```bash
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 -c 'import hashlib,json,sys; from bayesfilter.inference.hmc_serious_authority import AUTHORITY_PATH,CLAIM_PATH,INFRASTRUCTURE_FAILURE_PATH,INFRASTRUCTURE_MANIFEST_PATH,LOG_PATH,OUTPUT_MANIFEST_PATH,PRIVATE_SAMPLES_PATH,PROPOSAL_MANIFEST_PATH,PROPOSAL_PATH,PUBLIC_PROGRESS_PATH,PUBLIC_RESULT_PATH,SeriousInheritedEvidenceSession,_artifact_reference_from_snapshot,build_default_serious_authority_proposal,build_serious_authority_proposal_manifest,verify_serious_authority_proposal_candidate,verify_serious_authority_proposal_manifest; runtime_paths=(AUTHORITY_PATH,CLAIM_PATH,OUTPUT_MANIFEST_PATH,INFRASTRUCTURE_FAILURE_PATH,INFRASTRUCTURE_MANIFEST_PATH,PUBLIC_RESULT_PATH,PUBLIC_PROGRESS_PATH,PRIVATE_SAMPLES_PATH,LOG_PATH); present=[str(p) for p in runtime_paths if p.exists() or p.is_symlink()]; assert not present,present; evidence=SeriousInheritedEvidenceSession.open(extra_paths=(PROPOSAL_PATH,PROPOSAL_MANIFEST_PATH)); proposal_bytes=evidence.read_pinned_bytes(PROPOSAL_PATH); proposal=json.loads(proposal_bytes); manifest_bytes=evidence.read_pinned_bytes(PROPOSAL_MANIFEST_PATH); manifest=json.loads(manifest_bytes); config,preflight=verify_serious_authority_proposal_candidate(proposal,python_executable=sys.executable,evidence_session=evidence); rebuilt=build_default_serious_authority_proposal(python_executable=sys.executable,evidence_session=evidence); rebuilt_bytes=(json.dumps(rebuilt,sort_keys=True,indent=2)+"\n").encode(); assert rebuilt_bytes==proposal_bytes; verify_serious_authority_proposal_manifest(manifest,proposal_payload=proposal,proposal_bytes=proposal_bytes); reference=_artifact_reference_from_snapshot(path=PROPOSAL_PATH,payload=proposal,data=proposal_bytes); rebuilt_manifest=build_serious_authority_proposal_manifest(proposal_reference=reference); rebuilt_manifest_bytes=(json.dumps(rebuilt_manifest,sort_keys=True,indent=2)+"\n").encode(); assert rebuilt_manifest_bytes==manifest_bytes; evidence.verify_historical_archive_snapshot(); evidence.attempt1.verify_semantics(); evidence.verify(); evidence.close(); print(json.dumps({"status":"PASS_FINAL_ATTEMPT2_PRERUNTIME_STOP_GATE","proposal_artifact_hash":proposal["artifact_hash"],"proposal_file_sha256":hashlib.sha256(proposal_bytes).hexdigest(),"proposal_bytes":len(proposal_bytes),"manifest_artifact_hash":manifest["artifact_hash"],"manifest_file_sha256":hashlib.sha256(manifest_bytes).hexdigest(),"manifest_bytes":len(manifest_bytes),"runtime_authority":config.runtime_authority,"attempt2_runtime_paths_absent":len(runtime_paths),"attempt1_semantics":"verified","inherited_evidence":"verified"},sort_keys=True))'
```

Terminal output:

```text
{"attempt1_semantics": "verified", "attempt2_runtime_paths_absent": 9, "inherited_evidence": "verified", "manifest_artifact_hash": "sha256:64774b7c949386daf42d73291dbe2cccdc535625e92ab98ed349337c4d46e15e", "manifest_bytes": 869, "manifest_file_sha256": "e7aa19fb234dd3eff960e97c0c50a643c98663a6e87c98170a9c0f09c9a991b6", "proposal_artifact_hash": "sha256:e851b313f08e935f6bf4d67dca22448862e072dffc0fe32609580327e95182f4", "proposal_bytes": 39904, "proposal_file_sha256": "cb026193af3506719ecc17858979b4005b6a19a8eb2b8ad6d34a3800c60d0ab7", "runtime_authority": false, "status": "PASS_FINAL_ATTEMPT2_PRERUNTIME_STOP_GATE"}
```

TensorFlow also emitted CUDA registration and `cuInit` messages after CUDA was
deliberately hidden. Those messages are import noise, not GPU or runtime
evidence.

## Duplicate-Key Gate

Exit: `0`. Wall time: `<0.1 s`.

```bash
/home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 -c 'import json; from pathlib import Path
class DuplicateKey(ValueError): pass
def hook(pairs):
 d={}
 for k,v in pairs:
  if k in d: raise DuplicateKey(k)
  d[k]=v
 return d
for p in [Path("docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority_proposal.json"),Path("docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority_proposal_manifest.json")]:
 json.loads(p.read_text(),object_pairs_hook=hook); print("NO_DUPLICATE_KEYS",p)'
```

Both artifact paths printed `NO_DUPLICATE_KEYS`.

## Project Canonical-Hash Gate

Exit: `0`. Wall time: `3.9 s`.

```bash
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 -c 'import copy,hashlib,json; from pathlib import Path; from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash; from bayesfilter.inference.hmc_serious_authority import parse_serious_authority_proposal,parse_serious_authority_proposal_manifest; paths=[Path("docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority_proposal.json"),Path("docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority_proposal_manifest.json")];
for p in paths:
 x=json.loads(p.read_text()); y=copy.deepcopy(x); stated=y.pop("artifact_hash"); standard="sha256:"+hashlib.sha256(json.dumps(y,sort_keys=True,separators=(",",":")).encode()).hexdigest(); project=canonical_artifact_payload_hash(y); parsed=parse_serious_authority_proposal(x) if p.name.endswith("proposal.json") else parse_serious_authority_proposal_manifest(x); print(json.dumps({"path":str(p),"stated":stated,"project":project,"ordinary_compact":standard,"project_matches":stated==project,"parse_pass":parsed==x},sort_keys=True))'
```

Both `project_matches` and `parse_pass` values were `true`. The ordinary compact
hashes differed because they omit BayesFilter's declared type-tag normalization.

## Bound-File Hash Gate

Exit: `0`. Wall time: `<0.1 s`.

```bash
sha256sum bayesfilter/inference/hmc_serious_authority.py tests/test_hmc_serious_authority.py docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-attempt2-repair-subplan-2026-07-13.md docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-attempt1-infrastructure-result-2026-07-13.md docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-subplan-2026-07-11.md docs/plans/bayesfilter-hmc-semantic-identity-migration-phase6-smoke-result-2026-07-11.md docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority_proposal.json docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority_proposal_manifest.json
```

Observed hashes:

```text
4cb310f1845372c0857693f0e519d6b3f91b779d5502c30fb942e0716f1e2e29  bayesfilter/inference/hmc_serious_authority.py
58427c3d66dc7eb4fb9fb5694b5ebd2099419e093364170abb24655c49cdf201  tests/test_hmc_serious_authority.py
127b59b1a71c72fd8aedc6be8ca216a8e7386a7f76edbdcec4eb2bab1d516db3  docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-attempt2-repair-subplan-2026-07-13.md
b1e3c028e4121e04f9b29ab4bf2743f548fd8bc1d9bdb95514e967adffb1bd2b  docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-attempt1-infrastructure-result-2026-07-13.md
99fc680721acdb1a1d0502d91320f4459c186e4050e49d38a4cbf9b75d480be9  docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-subplan-2026-07-11.md
c8253351c7ff01f844256bfee4d3f6fe820ba86cf6c6c32c1d7c3fa5db78305a  docs/plans/bayesfilter-hmc-semantic-identity-migration-phase6-smoke-result-2026-07-11.md
cb026193af3506719ecc17858979b4005b6a19a8eb2b8ad6d34a3800c60d0ab7  docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority_proposal.json
e7aa19fb234dd3eff960e97c0c50a643c98663a6e87c98170a9c0f09c9a991b6  docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority_proposal_manifest.json
```

## Process-Absence Gate

Exit: `1`, meaning no filtered match. Wall time: `<0.1 s`.

```bash
ps -ef | rg '[r]un_hmc_phase7_typed_identity_serious|[p]hase7_serious_attempt2|[d]eterministic_lgssm_hmc_phase7_tf'
```

## Diagnostic History

The first exact-reconstruction command exited `1` after `9.8 s` because it
asserted Python object equality between JSON-loaded arrays and builder tuples.
A localization command exited `0` after `9.1 s`, identifying only `command` and
`nonclaims` as object-type differences while proving identical proposal and
manifest bytes. The final decision-bearing command above uses exact serialized
bytes plus strict parsers and evidence-session semantics.

## Preserved Artifacts

- `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority_proposal.json`
- `docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_attempt2_authority_proposal_manifest.json`
- `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-attempt2-preruntime-result-2026-07-13.md`
- `docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase7-attempt2-proposal-codex-review-2026-07-13.md`
- `docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase7-attempt2-manifest-codex-review-2026-07-13.md`
- `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-serious-attempt2-runtime-subplan-2026-07-13.md`
- `docs/reviews/bayesfilter-hmc-semantic-identity-migration-phase7-attempt2-runtime-subplan-codex-review-2026-07-13.md`
