"""Terminal diagnostic audit; no estimator or sampler default authority."""
from pathlib import Path
import gzip,hashlib,importlib.util,json,math,sys,time
from datetime import datetime,timezone
root=Path(__file__).resolve().parent
started=time.monotonic()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normalized(x):return json.loads(json.dumps(x,sort_keys=True))
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module(root/"calibration-driver.py","old_calibration")
new=module(root/"calibration-analysis.py","new_calibration")
supervision=read(root/"calibration-supervision-01/result.json")
if supervision["status"]!="completed":raise ValueError("terminal complete calibration required")
terminal=Path(supervision["terminal_result"])
report=read(terminal)
if report["status"]!="completed" or report["seed"]!=73113 or report["replications"]!=128:
    raise ValueError("wrong completion/design")
if report["profile"]!="batch_maximum_repair" or report["methods"]!=list(old.REPAIR_METHODS):
    raise ValueError("wrong profile")
if report["chains"]!=4 or report["looks"]!=list(old.LOOKS):raise ValueError("wrong schedule")
if [(c["law"],c["rho"]) for c in report["cells"]]!=[(law,rho) for rho in old.RHOS for law in ("gaussian","student_t5")]:
    raise ValueError("wrong condition inventory")
for name,digest in report["source_sha256"].items():
    if sha(Path(name))!=digest:raise ValueError("changed numerical source: "+name)
if sha(root/"calibration-driver.py")!=report["driver_sha256"]:raise ValueError("changed frozen driver")
carried=0
if "resume_from" in report:
    prior=Path(report["resume_from"]["result_file"])
    if sha(prior)!=report["resume_from"]["result_sha256"]:raise ValueError("changed continuation parent")
    previous=read(prior)
    for a,b in zip(previous["cells"],report["cells"]):
        count=len(a["records"])
        if a["records"]!=b["records"][:count]:raise ValueError("changed carried records")
        carried+=count
cells=[];inspected_endpoints=0
for index,cell in enumerate(report["cells"]):
    if len(cell["records"])!=128 or cell["planned"]!=128:raise ValueError("incomplete denominator")
    if normalized(old.summarize(cell["records"],128,old.REPAIR_METHODS))!=cell["summary"]:
        raise ValueError("stored method summary mismatch")
    if normalized(old.summarize_pairs(cell["records"]))!=cell["paired_summary"]:
        raise ValueError("stored legacy paired summary mismatch")
    for replication,record in enumerate(cell["records"]):
        if record["replication"]!=replication or record["seed"]!=[73113,index//2,replication]:
            raise ValueError("replication identity mismatch")
        if set(record["methods"])!=set(old.REPAIR_METHODS):raise ValueError("missing method")
        for name,row in record["methods"].items():
            if row["fixed"] is None or row["fixed"]["draws"]!=10000:raise ValueError("missing fixed endpoint")
            for kind in ("fixed","stopped"):
                endpoint=row[kind]
                if endpoint is None:continue
                inspected_endpoints+=1
                if endpoint["draws"] not in old.LOOKS or (kind=="stopped" and not endpoint["passed"]):
                    raise ValueError("invalid stopping record")
                for est,truth,err,se,valid,miss in zip(endpoint["estimate"],record["truth"],endpoint["error"],endpoint["mcse"],endpoint["valid"],endpoint["interval_miss"]):
                    if not math.isclose(est-truth,err,rel_tol=1e-12,abs_tol=1e-14):raise ValueError("changed error")
                    if valid:
                        if se is None or not math.isfinite(se) or se<=0 or miss!=(abs(err)>1.959963984540054*se):
                            raise ValueError("invalid interval arithmetic")
                    elif se is not None or miss is not None:raise ValueError("unavailable interval hidden")
        guard=record["methods"]["lugsail_max_batch_means"]
        for kind in ("fixed","stopped"):
            endpoint=guard[kind]
            if endpoint is None:continue
            meta=endpoint["batch_metadata"]
            B=meta["per_chain_plain_long_run_variance"];L=meta["per_chain_raw_lugsail_long_run_variance"]
            M=meta["per_chain_long_run_variance"];mask=meta["per_chain_batch_maximum_activated"]
            for chain in range(4):
                for q in range(3):
                    b,l,v=B[chain][q],L[chain][q],M[chain][q]
                    ok=b is not None and l is not None and b>0
                    if v!=(max(b,l) if ok else None) or mask[chain][q]!=(ok and b>l):
                        raise ValueError("maximum or activation mismatch")
            for q in range(3):
                available=all(M[c][q] is not None and M[c][q]>0 for c in range(4))
                if endpoint["valid"][q]!=available:raise ValueError("availability mismatch")
                if available and not math.isclose(endpoint["mcse"][q]**2,sum(M[c][q] for c in range(4))/(16*endpoint["draws"]),rel_tol=1e-12,abs_tol=1e-15):
                    raise ValueError("pooling mismatch")
        a=record["methods"]["lugsail"]["fixed"];b=guard["fixed"]
        if a["estimate"]!=b["estimate"]:raise ValueError("changed fixed-endpoint estimate")
        for x,y,valid in zip(a["mcse"],b["mcse"],a["valid"]):
            if valid and (y is None or y+1e-14<x):raise ValueError("violated maximum inequality")
    activations={kind:sum(sum(sum(row) for row in r["methods"]["lugsail_max_batch_means"][kind]["batch_metadata"]["per_chain_batch_maximum_activated"]) for r in cell["records"] if r["methods"]["lugsail_max_batch_means"][kind] is not None) for kind in ("fixed","stopped")}
    paired=new.summarize_pairs(cell["records"],law=cell["law"])
    cells.append(dict(law=cell["law"],rho=cell["rho"],summary=cell["summary"],
        paired_summary=paired,activations=activations))
archives=[]
for attempt in supervision["attempts"]:
    f=Path(attempt["output"])/"result.json";raw=f.read_bytes();archive=f.with_suffix(".json.gz")
    if archive.exists():raise ValueError("archive already exists")
    archive.write_bytes(gzip.compress(raw,mtime=0))
    if gzip.decompress(archive.read_bytes())!=raw:raise ValueError("archive roundtrip failed")
    archives.append(dict(file=str(f),sha256=sha(f),archive=str(archive),archive_sha256=sha(archive)))
sensitivity=read(root/"mams-sensitivity-01-supervisor.json")
if sensitivity["returncode"]!=0:raise ValueError("sensitivity failure")
result=dict(status="passed",terminal_result=str(terminal),terminal_sha256=sha(terminal),
    calibration_supervision_sha256=sha(root/"calibration-supervision-01/result.json"),
    corrected_analysis_sha256=sha(root/"calibration-analysis.py"),driver_sha256=sha(Path(__file__)),
    command=[sys.executable,*sys.argv],environment=sys.executable,
    plan="docs/plans/bayesfilter-modern-hmc-comparator-precision-continuation-2026-10-13.md",
    source_hash_count=len(report["source_sha256"]),law_records=768,independent_latent_units=384,
    carried_records_unchanged=carried,endpoints_inspected=inspected_endpoints,
    interpretation="fixed endpoint and policy-stopped intervals; no anytime coverage or default promotion",
    replaced_legacy_binary_student_t_intervals=True,method_ranking=None,default_promotion=False,
    original_raw_records_unchanged=True,archives=archives,cells=cells,
    calibration_supervisor_seconds=supervision["elapsed_seconds"],sensitivity_supervisor_seconds=sensitivity["elapsed_seconds"])
result["audit_seconds"]=time.monotonic()-started
result["cpu_reservation_charge_seconds"]=supervision["elapsed_seconds"]+sensitivity["elapsed_seconds"]+result["audit_seconds"]
if result["cpu_reservation_charge_seconds"]>3600:raise ValueError("CPU reservation overspent")
out=root/"terminal-audit-01";out.mkdir(exist_ok=False)
(out/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:result[k] for k in ("status","law_records","endpoints_inspected","audit_seconds","cpu_reservation_charge_seconds")}))
