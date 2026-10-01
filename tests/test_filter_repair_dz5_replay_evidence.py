"""Independent saved-evidence checks; no target execution or admission."""

import importlib.util
import json
from pathlib import Path


def test_dz5_replay_saved_evidence(request):
    path = Path(__file__).resolve().parents[1] / ('docs/plans/artifacts/'
        'filter-gradient-repair-20260917/analyze-dz5-replay-20260925.py')
    spec = importlib.util.spec_from_file_location('dz5_replay_analysis', path)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    result = analyzer.analyze(Path('/home/ubuntu/workspace/BayesFilter/docs/plans/'
        'artifacts/filter-gradient-repair-20260917'))
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / 'dz5-replay-analysis.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n')
