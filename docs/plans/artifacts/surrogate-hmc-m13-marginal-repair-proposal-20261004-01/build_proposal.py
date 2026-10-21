"""Build and inspect the proposal. CPU document work; no algorithm imports."""
import hashlib,json,pathlib,subprocess,time
from PIL import Image,ImageOps,ImageDraw
root=pathlib.Path.cwd()
out=pathlib.Path(__file__).resolve().parent
tex=root/"docs/plans/surrogate-hmc-m13-marginal-repair-proposal-2026-10-04.tex"
cmd=["/usr/bin/pdflatex","-interaction=nonstopmode","-halt-on-error",
     "-output-directory="+str(out),str(tex)]
start=time.monotonic()
codes=[]
with (out/"latex-build.log").open("w") as log:
    for _ in range(2):
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        codes.append(p.returncode)
        if p.returncode: break
report={"command":cmd,"exit_codes":codes,
        "wall_seconds":time.monotonic()-start,
        "source_sha256":hashlib.sha256(tex.read_bytes()).hexdigest(),
        "initial_attempt":"latexmk not installed; used installed pdflatex"}
(out/"latex-build-manifest.json").write_text(json.dumps(report,indent=2)+"\n")
if any(codes):
    print((out/"latex-build.log").read_text()[-1800:])
    raise SystemExit(1)
log=(out/(tex.stem+".log")).read_text()
warnings=[l for l in log.splitlines()
          if any(t in l for t in ["Overfull","Underfull","Warning","undefined","multiply"])]
pdf=out/(tex.stem+".pdf")
subprocess.run(["pdftotext","-layout",str(pdf),str(out/"proposal-rendered.txt")],check=True)
subprocess.run(["pdftoppm","-scale-to","1100","-png",str(pdf),str(out/"proposal-page")],
               check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
pages=sorted(out.glob("proposal-page-*.png"))
for first in range(0,len(pages),3):
    chosen=pages[first:first+3]
    thumb_w=620
    thumbs=[]
    for p in chosen:
        im=Image.open(p).convert("RGB")
        im.thumbnail((thumb_w,900))
        thumbs.append(im)
    sheet=Image.new("RGB",(3*thumb_w+40,950),"#d0d0d0")
    draw=ImageDraw.Draw(sheet)
    for j,im in enumerate(thumbs):
        sheet.paste(im,(10+j*(thumb_w+10),30))
        draw.text((15+j*(thumb_w+10),8),chosen[j].name,fill="black")
    sheet.save(out/f"contact-{first//3+1}.jpg",quality=88)
print(json.dumps({"pdf":str(pdf),"pages":len(pages),"warnings":warnings,
                  "contact_sheets":(len(pages)+2)//3,"build":report}))
