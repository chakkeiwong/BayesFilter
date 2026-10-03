"""Offline phase audit; no numerical experiments or GPU initialization."""
import hashlib,json,os,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[4]
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
    started=time.monotonic()
    sys.path.insert(0,str(ROOT/"source-final-r2"))
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
    from bayesfilter.inference.hmc_candidate_set_artifacts import load_candidate_set_result_payload
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    invalid=[]; attempts=[]; costs={"cpu_reference":0.,"gpu":0.}; snapshots=[]
    for p in sorted(ROOT.glob("*-run.json"))+sorted(ROOT.glob("*/gpu-diagnostic-run.json")):
        d=read(p); costs[d["device"]]+=d["elapsed_seconds"]
        attempts.append({"path":str(p.relative_to(REPO)),"sha256":sha(p),"seconds":d["elapsed_seconds"],"returncode":d["returncode"],"device":d["device"]})
        if d["device"]=="gpu":
            rt=d["runtime"]; mem=rt["memory_policy"]
            if not (rt["jit_compile"] and mem["all_physical_devices_memory_growth"] and mem["configured_before_logical_device_initialization"]): invalid.append(str(p)+": runtime provenance")
        elif d.get("gpu_intentionally_hidden") is not True: invalid.append(str(p)+": CPU provenance")
    matrix=read(ROOT/"public-model-matrix-gpu-r1/run_index.json")
    matrix_status={}
    for name,job in matrix["jobs"].items():
        matrix_status[name]=job["status"]
        for a in job["attempts"]:
            costs["gpu"]+=a["elapsed_seconds"]
            attempts.append({"path":a["log"],"seconds":a["elapsed_seconds"],"device":"gpu","status":a["status"]})
    for p in sorted(ROOT.glob("source*/source_snapshot.json")):
        d=read(p); actual={str(f.relative_to(p.parent)):sha(f) for f in sorted((p.parent/"bayesfilter").rglob("*.py"))}
        if actual!=d["source_files"] or _sha256(actual)!=d["source_identity"]: invalid.append(str(p)+": snapshot checksum")
        snapshots.append({"path":str(p.relative_to(REPO)),"identity":d["source_identity"],"different_from_current":[f for f,h in actual.items() if sha(REPO/f)!=h]})
    inventories=[]; receipts=0
    for p in sorted(ROOT.glob("*/tuning/candidate_set_result.json"))+sorted(ROOT.glob("public-model-matrix-gpu-r1/*/replication-*/tuning/candidate_set_result.json")):
        d=load_candidate_set_result_payload(p); inv=check_inventory(d); invalid.extend(inv["failures"])
        inventories.append({"path":str(p.relative_to(REPO)),**inv})
        for rec in d["verification_receipts"]:
            e=read(p.parent/"numerical_evidence"/(rec["numerical_evidence_hash"]+".json"))
            if _sha256(e)!=rec["numerical_evidence_hash"] or e["candidate"]["candidate_record_hash"]!=rec["candidate_record_hash"]: invalid.append(str(p)+": receipt")
            receipts+=1
    tensors=0
    for p in ROOT.rglob("*.tensor.json"):
        if sha(p.with_suffix(""))!=read(p)["sha256"]: invalid.append(str(p)+": tensor")
        tensors+=1
    def check(node,base):
        nonlocal tensors
        if isinstance(node,dict):
            if "tensor" in node:
                d=node["tensor"]; tensors+=1
                if sha(base/d["file"])!=d["sha256"]: invalid.append(str(base/d["file"])+": checkpoint tensor")
            else:
                for v in node.values(): check(v,base)
        elif isinstance(node,list):
            for v in node: check(v,base)
    for p in ROOT.glob("*/posterior_chunks/committed/*/bundle.json"): check(read(p)["tree"],p.parent)
    latest={}
    for p in sorted(ROOT.glob("*.xml"),key=lambda p:p.stat().st_mtime):
        for case in ET.parse(p).getroot().iter("testcase"):
            status="failed" if case.find("failure") is not None or case.find("error") is not None else ("skipped" if case.find("skipped") is not None else "passed")
            latest[case.attrib["classname"]+"::"+case.attrib["name"]]={"status":status,"source":p.name}
    invalid.extend(k for k,v in latest.items() if v["status"]=="failed")
    a=ROOT/"inactive-option-parity-gpu-r1"; b=ROOT.parent/"m12-r1/inactive-option-parity-gpu-r1"
    parity={p.name:sha(p)==sha(b/p.name) for p in a.glob("*.tensor")}
    da,db=read(a/"assessment.json"),read(b/"assessment.json")
    parity.update({k:da[k]==db[k] for k in ("final_epsilon","final_transform")})
    if not all(parity.values()): invalid.append("inactive option parity")
    guide=read(ROOT/"guide-r1/build-manifest.json")
    if sha(REPO/"docs/main.pdf")!=guide["pdf_sha256"]: invalid.append("installed book checksum")
    if any(sha(REPO/f)!=h for f,h in guide["source_sha256"].items()): invalid.append("book source changed")
    opening=read(ROOT.parent/"m12-r1/reconciliation-terminal.json")["budget_seconds"]
    overhead={"cpu_reference":900.,"gpu":0.}
    total={k:costs[k]+overhead[k] for k in costs}; charged={k:opening["cumulative_charged"][k]+total[k] for k in costs}
    remaining={k:opening["authorized"][k]-charged[k] for k in costs}
    budget={"authorized":opening["authorized"],"opening_charged":opening["cumulative_charged"],"measured":costs,"overhead":overhead,"phase_total":total,"cumulative_charged":charged,"remaining":remaining,"exceeded":total["cpu_reference"]>6000 or total["gpu"]>9000 or min(remaining.values())<0}
    d={"schema":"bayesfilter.hmc_m13_reconciliation.v1","terminal":True,"source_snapshots":snapshots,"attempts":attempts,"candidate_inventories":inventories,"numerical_receipts_checked":receipts,"tensor_checksums_checked":tensors,"tests":{s:sum(v["status"]==s for v in latest.values()) for s in ("passed","failed","skipped")},"test_details":latest,"inactive_option_parity":parity,"matrix_status":matrix_status,"invalid_artifacts":invalid,"outstanding_workers":[],"budget_seconds":budget,"manifest":{"command":sys.argv,"environment":sys.executable,"script_sha256":sha(Path(__file__)),"elapsed_seconds":time.monotonic()-started,"gpu_intentionally_hidden":True},"interpretation":"Integrity and accounting only; matrix timeouts remain open resource repairs."}
    with (ROOT/"reconciliation-terminal.json").open("x") as f: json.dump(d,f,indent=2,allow_nan=False); f.write("\n")
    print(json.dumps({k:d[k] for k in ("tests","budget_seconds","invalid_artifacts","matrix_status")},indent=2))
    return 1 if invalid or budget["exceeded"] else 0
if __name__=="__main__": sys.exit(main())
