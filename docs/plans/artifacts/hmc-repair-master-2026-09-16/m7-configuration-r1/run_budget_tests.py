"""Run the specifically affected preset, budget and translation tests."""
from pathlib import Path
import json
import subprocess
import sys
root=Path(__file__).resolve().parent
selection=json.loads((root/'budget-test-selection.json').read_text())
command=[sys.executable,'-m','pytest',*selection,'-q','--disable-warnings','--maxfail=3',
         '--junitxml='+str(root/'budget-tests.xml')]
raise SystemExit(subprocess.run(command,cwd=root.parents[4]).returncode)
