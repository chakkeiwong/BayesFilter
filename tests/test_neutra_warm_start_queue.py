"""CPU-only supervisor tests; no model training or scientific conclusions."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('warm_start_queue',ROOT/'scripts/continue_neutra_warm_start_campaign.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


def test_queue_uses_existing_budget_root_and_resumes_without_duplicate_commands(tmp_path):
    (tmp_path/'config.json').write_text('{}')
    calls=[]
    def execute(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=0)
    with patch.object(queue,'steps',return_value=[('first',['--through','train']),('second',['--through','qualify'])]):
        assert queue.run_queue(tmp_path,execute=execute)==0
        assert queue.run_queue(tmp_path,execute=execute)==0
    assert len(calls)==2
    assert all(cmd[cmd.index('--output')+1]==str(tmp_path) for cmd in calls)
    state=json.loads((tmp_path/'continuation-queue-state.json').read_text())
    assert state['status']=='queue_attempted'
    assert 'individual results' in state['interpretation']


def test_queue_stops_on_master_failure_and_preserves_failed_command_for_retry(tmp_path):
    (tmp_path/'config.json').write_text('{}');calls=[]
    def execute(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(returncode=3)
    with patch.object(queue,'steps',return_value=[('first',[]),('dependent',[])]):
        assert queue.run_queue(tmp_path,execute=execute)==3
    state=json.loads((tmp_path/'continuation-queue-state.json').read_text())
    assert len(calls)==1 and state['status']=='budget_exhausted'
    assert state['steps']['first']['status']=='stopped' and 'dependent' not in state['steps']
