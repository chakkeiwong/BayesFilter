"""Bounded retrieval of foundational sources cited by the monograph chapter."""
from pathlib import Path
from urllib.request import Request, urlopen
import json

dest = Path("/home/ubuntu/python/BayesFilter/.localresources/hmc-methods-review-20261007")
sources = [
    ("neal-hmc-2011", "https://arxiv.org/pdf/1206.1901"),
    ("nuts-2014", "https://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf"),
    ("batch-means-2010", "https://arxiv.org/pdf/0811.1729"),
    ("particle-mcmc-2010", "https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf"),
]
results = []
for name, url in sources:
    path = dest / (name + ".pdf")
    try:
        if path.exists():
            results.append({"name": name, "url":url, "status":"already_present"})
            continue
        with urlopen(Request(url, headers={"User-Agent":"BayesFilter-literature-review/1.0"}),
                     timeout=25) as response:
            data = response.read()
        if not data.startswith(b"%PDF"):
            raise ValueError("Response is not a PDF")
        path.write_bytes(data)
        results.append({"name":name,"url":url,"status":"downloaded","bytes":len(data)})
    except Exception as exc:
        results.append({"name":name,"url":url,"status":"failed","error":str(exc)})
    print(json.dumps(results[-1]),flush=True)
Path(__file__).with_name("foundational-retrieval.json").write_text(json.dumps(results,indent=2)+"\n")
