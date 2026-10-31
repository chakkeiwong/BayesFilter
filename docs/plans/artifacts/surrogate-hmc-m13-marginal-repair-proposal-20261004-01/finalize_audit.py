"""Finalize provenance for the mathematical proposal; no runtime imports."""
import datetime,hashlib,json,pathlib,platform,re,subprocess,sys
root=pathlib.Path.cwd()
out=pathlib.Path(__file__).resolve().parent
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((out/"runtime-source-before.json").read_text())
after={n:digest(root/n) for n in before if (root/n).exists()}
changed=[n for n in before if before[n]!=after.get(n)]
new_sources=sorted(set(str(p.relative_to(root)) for p in
    list((root/"bayesfilter").rglob("*.py"))+
    list((root/"experiments/dpf_implementation/tf_tfp/fixtures").glob("*.py")))-set(before))
(out/"runtime-source-after.json").write_text(json.dumps(after,indent=2)+"\n")
sources={}
for label,texname,reportname in [
    ("v1","proposal-audited-v1.tex","mathdev-document-rigor.json"),
    ("v2","proposal-audited-v2.tex","mathdev-revision-rigor.json")]:
    d=json.loads((out/reportname).read_text())
    actual=digest(out/texname)
    expected=d["source"]["source_digest"]
    sources[label]={"snapshot":texname,"sha256":actual,"report":reportname,
                    "report_source_sha256":expected,"matches":actual==expected,
                    "coverage":d["coverage"]}
tex=root/"docs/plans/surrogate-hmc-m13-marginal-repair-proposal-2026-10-04.tex"
s=tex.read_text()
labels=re.findall(r"\\label\{([^}]+)\}",s)
refs=re.findall(r"\\(?:eqref|ref)\{([^}]+)\}",s)
missing=sorted(set(refs)-set(labels))
duplicate=sorted({l for l in labels if labels.count(l)>1})
algebra=json.loads((out/"local-algebra-report.json").read_text())
build=json.loads((out/"latex-build-manifest.json").read_text())
checks={
    "runtime_source_files_unchanged":not changed and not new_sources,
    "audited_source_digests_match":all(d["matches"] for d in sources.values()),
    "local_algebra_passed":algebra["status"]=="PASS",
    "pdf_build_passed":build["exit_codes"]==[0,0],
    "pdf_matches_final_source":build["source_sha256"]==digest(tex),
    "all_equation_references_resolve":not missing,
    "equation_labels_unique":not duplicate,
    "claude_handoff_exists":(root/"docs/plans/surrogate-hmc-m13-marginal-repair-claude-handoff-2026-10-04.md").is_file(),
}
final={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
       "runtime_files_checked":len(before),"changed_runtime_files":changed,
       "new_runtime_files":new_sources,"missing_references":missing,
       "duplicate_labels":duplicate,
       "render_review":"All nine pages inspected via three contact sheets; equations/table fit, no clipping or blank pages.",
       "mathematical_scope":"Scoped identities and explicit manual proofs; no full formal certificate or empirical candidate run."}
(out/"final-artifact-check.json").write_text(json.dumps(final,indent=2)+"\n")
commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip()
status=subprocess.check_output(["git","status","--short","--untracked-files=no"],text=True)
source_pdfs={str(p.relative_to(root)):digest(p) for p in
             (root/".localresources/papers/m13-repair-20261004").glob("*.pdf")}
manifest={
 "schema":"bayesfilter_research_proposal_audit_v1",
 "completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "question":"Mathematically valid M13 importance correction repair while preserving original LGSSM/KSC values and scores; downstream goal remains surrogate HMC.",
 "stage":"proposal_and_math_audit_only",
 "git_commit":commit,"git_branch":branch,"tracked_worktree_status":status.splitlines(),
 "source_baseline":"Current dirty post-fix checkout; runtime-source-before.json is the exact file hash snapshot.",
 "runtime_source_files_checked":len(before),"runtime_source_changed":bool(changed or new_sources),
 "python":sys.executable,"python_version":platform.python_version(),
 "conda_environment":"tftwogpu",
 "execution":"CPU-only symbolic/reference and document work. No TensorFlow/CUDA import, GPU allocation, filter run or HMC.",
 "gpu_memory_policy":"N/A: no GPU workload",
 "jit_compile":"N/A: no TensorFlow algorithm executed",
 "random_seeds":"N/A: exact deterministic algebra; empirical inputs belong to linked previous artifacts",
 "data_version":"No new data; October 4 localization and score-before/after artifacts are cited as prior evidence.",
 "plan_file":"docs/plans/surrogate-hmc-m13-marginal-repair-audit-plan-2026-10-04.md",
 "result_file":str((out/"audit-result.md").relative_to(root)),
 "proposal_file":str(tex.relative_to(root)),"proposal_sha256":digest(tex),
 "commands":[
  "/home/chakwong/anaconda3/envs/tftwogpu/bin/python "+str((out/"verify_algebra.py").relative_to(root)),
  "/home/chakwong/anaconda3/envs/tftwogpu/bin/python "+str((out/"build_proposal.py").relative_to(root)),
  "/home/chakwong/anaconda3/envs/tftwogpu/bin/python "+str((out/"finalize_audit.py").relative_to(root))],
 "document_build_command":build["command"],
 "wall_time_seconds":{"local_algebra":algebra["wall_seconds"],"final_pdf_compile":build["wall_seconds"],
                      "mcp_document_call_timeout_each":300},
 "budget":{"local_algebra_limit_seconds":120,"gpu_hmc_seconds_used":0,
           "previous_diagnostic_balance_seconds":1575.90,
           "service_latency_and_document_builds":"excluded from local algebra budget"},
 "mathdev":{"primary_equivalent_checks":11,"additional_unreduced_gaussian_coefficients":5,
            "local_checks":len(algebra["checks"]),"document_audit":"partial; original and narrowed reports retained",
            "formal_certificate":False,
            "raw_results":["mathdev-symbolic-raw.json","mathdev-supplemental-raw.json"]},
 "audited_sources":sources,"source_pdf_sha256":source_pdfs,
 "artifacts":{p.name:{"sha256":digest(p),"bytes":p.stat().st_size}
              for p in sorted(out.iterdir()) if p.is_file() and p.name!="run-manifest.json"},
 "final_artifact_check":final,
 "claude_review":"Written handoff only; not yet performed",
 "next_master_phase":"Implement separately identified optional marginal correction, preserve legacy consumers, then test T=1 against bootstrap/current LEDH/quadrature; proposal only."
}
(out/"run-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"status":final["status"],"checks":checks,"runtime_files":len(before),
                  "changed_runtime_files":changed,"source_digest_match":{k:v["matches"] for k,v in sources.items()},
                  "manifest":str((out/"run-manifest.json").relative_to(root))}))
if not all(checks.values()): raise SystemExit(1)
