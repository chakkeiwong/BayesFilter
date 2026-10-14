"""CPU-only inspection of rejected Lorenz safeguard histories; no candidate selection."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import json
import subprocess
import numpy as np
import reference_lorenz_stable_fit as independent
from pathlib import Path
import time
import tensorflow as tf

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("output","candidate","reference","design"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--mode")
    args=p.parse_args();start=time.monotonic()
    source=json.loads((args.candidate/"summary.json").read_text())
    summary=dict(status="PASS",device="CPU",gpu_intentionally_hidden=True,
        role="explanatory_rejected_candidate_history",replication="INCOMPLETE",checks=[],candidates=[])
    for candidate in source["candidates"]:
        rows={};history={}
        for key in ("A","b","c","x","log_incremental_weight","logZ","ess","precision_margin"):
            value=tf.io.parse_tensor(tf.io.read_file(str(args.candidate/candidate["folder"]/(key+".tensor"))),tf.float64)
            history[key]=value
            finite=tf.math.is_finite(value);bad=tf.where(~finite)
            rows[key]=dict(shape=value.shape.as_list(),nonfinite_count=int(tf.shape(bad)[0].numpy()),
                first_nonfinite=bad[0].numpy().tolist() if int(tf.shape(bad)[0].numpy()) else None,
                maximum_finite_magnitude=float(tf.reduce_max(tf.where(finite,tf.abs(value),0.)).numpy()))
        item=dict(floor=candidate["floor"],activation_margin=source.get("activation_margin"),fields=rows)
        bad=rows["x"]["first_nonfinite"] or rows["log_incremental_weight"]["first_nonfinite"]
        if bad:
            stage,t,row=bad[:3]
            # Preserve the first failing transition inputs for a subsequent independent check.
            inspected=[]
            for index in range(max(0,t-2),min(history["x"].shape[1],t+2)):
                value=history["x"][stage,index,row]
                finite=tf.math.is_finite(value)
                inspected.append(dict(time=index,nonfinite=int(tf.reduce_sum(tf.cast(~finite,tf.int32)).numpy()),
                    max_abs_finite=float(tf.reduce_max(tf.where(finite,tf.abs(value),0.)).numpy())))
                tf.io.write_file(str(args.output/("stage%d_row%d_time%d.tensor"%(stage,row,index))),tf.io.serialize_tensor(value))
            item.update(first_bad_stage=stage,first_bad_time=t,first_bad_row=row,nearby_clouds=inspected)

        if args.mode=="transition" and bad:
            # Same two clouds through an independent equation and unchanged author RK4.
            at=max(0,t-1)
            values=np.stack([history["x"][stage-1,at,row].numpy(),history["x"][stage,at,row].numpy()])
            with np.errstate(over="ignore",invalid="ignore"):
                expected=independent.rk4(values,2.5)
            input_path=args.output/"transition-input.bin"
            input_path.write_bytes(values.astype("<f8").tobytes())
            output_path=args.output/"author-transition.bin"
            author_path=Path(__file__).resolve().parent
            quote=lambda path:str(path).replace("'","''")
            command_text=("addpath('"+quote(author_path)+"');"+
                "fid=fopen('"+quote(input_path)+"','rb'); X=reshape(fread(fid,Inf,'double'),8,[])'; fclose(fid);"+
                "d=8; P=diag(ones(1,d-1),-1); P(1,d)=1; PT=P'; PP=P*P;"+
                "Lorenz=@(x) ((PT*x'-PP*x').*(P*x')-x'+2.5)';"+
                "Y=RungeKutta4(X,0.1,Lorenz);"+
                "fid=fopen('"+quote(output_path)+"','wb'); fwrite(fid,Y','double'); fclose(fid);")
            matlab=args.output/"author-transition.m";matlab.write_text(command_text+"\n")
            command=["octave","--no-gui","--quiet",str(matlab)]
            with (args.output/"author-transition.log").open("w") as log:
                subprocess.run(command,check=True,timeout=8,stdout=log,stderr=subprocess.STDOUT)
            actual=np.frombuffer(output_path.read_bytes(),dtype="<f8").reshape(expected.shape)
            mask=np.isfinite(expected)
            same_mask=bool(np.array_equal(np.isfinite(actual),mask))
            finite_parity=bool(np.all(np.abs(actual[mask]-expected[mask])<=1e-10+1e-12*np.abs(expected[mask])))
            original_nonfinite=[int(np.sum(~np.isfinite(actual[i]))) for i in range(2)]
            reference_nonfinite=[int(np.sum(~np.isfinite(expected[i]))) for i in range(2)]
            item["transition_reference"]=dict(command=command,author_source="RungeKutta4.m:12-21; cSMC_VaryF.m:11-16,94",
                source_stages=[stage-1,stage],time=at,author_nonfinite=original_nonfinite,
                independent_nonfinite=reference_nonfinite,same_finiteness_mask=same_mask,
                finite_values_agree=finite_parity,diagnostic_numpy_reference=True)
            summary["checks"].append(dict(name="author_and_independent_transition_agreement",
                passed=same_mask and finite_parity and original_nonfinite[0]==0 and original_nonfinite[1]>0))
        summary["candidates"].append(item)
    summary["status"]="PASS" if all(row["passed"] for row in summary["checks"]) else "FAIL"
    summary["wall_seconds"]=time.monotonic()-start
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary))
if __name__=="__main__":main()
