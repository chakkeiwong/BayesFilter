"""Read-only Codex incident measurements; never export raw prompts or tool output.

Run with a session UUID. Writes sanitized JSON beside this script. Byte and
character counts are storage/traffic measures, not active-context token counts.
Only top-level response items are counted, avoiding event/completion duplicates.
"""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent
CODEX = Path('/home/chakwong/.codex')


def text_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(text_content(x) for x in value)
    if isinstance(value, dict):
        return text_content(value.get('text', value.get('content', '')))
    return ''


def output_text(value):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return value
        if isinstance(parsed, (dict, list)):
            return output_text(parsed)
        return value
    if isinstance(value, list):
        return '\n'.join(output_text(x) for x in value)
    if isinstance(value, dict):
        for key in ['output', 'text', 'content', 'result']:
            if key in value:
                return output_text(value[key])
        return ''
    return ''


def tool_stats(rows):
    return dict(count=len(rows), text_chars=sum(r['text_chars'] for r in rows),
                over_8000_chars=sum(r['text_chars'] > 8000 for r in rows),
                over_20000_chars=sum(r['text_chars'] > 20000 for r in rows),
                truncation_notices=sum(r['truncated'] for r in rows),
                largest_requested_budget=max((r['max_budget'] for r in rows), default=0))


def history_stats(history):
    sizes = Counter()
    hashes = Counter()
    entries = []
    for item in history or []:
        txt = text_content(item.get('content', item.get('output', item.get('summary', ''))))
        kind = item.get('role', item.get('type', '?'))
        sizes[kind] += len(txt)
        if txt:
            hashes[sha256(txt.encode()).hexdigest()] += 1
        entries.append({'kind': kind, 'text_chars': len(txt),
                        'global_policy_headers': txt.count('# Global Scientific Coding Agent Policy'),
                        'context_policy_headers': txt.count('## Context, Tool Output, And Recovery Discipline')})
    return {'items': len(history or []), 'text_chars_by_kind': dict(sizes),
            'serialized_chars': len(json.dumps(history, ensure_ascii=False)),
            'duplicate_nonempty_copies': sum(n-1 for n in hashes.values()),
            'largest_entries': sorted(entries, key=lambda x: x['text_chars'], reverse=True)[:8],
            'global_policy_headers': sum(x['global_policy_headers'] for x in entries)}


def analyze(sid):
    conn = sqlite3.connect((CODEX/'state_5.sqlite').as_uri()+'?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    state = dict(conn.execute('SELECT id,rollout_path,cwd,model,model_provider,cli_version,created_at,updated_at FROM threads WHERE id=?', (sid,)).fetchone())
    conn.close()
    path = Path(state['rollout_path'])
    calls, outputs, messages, turns, compactions, tokens, model_changes = {}, [], [], [], [], [], []
    kinds = Counter()
    segment = 0
    turn = None
    source_bytes = 0
    source_hash = sha256()
    context_shape = None
    meta = {}
    with path.open('rb') as stream:
        for line, raw in enumerate(stream, 1):
            source_bytes += len(raw)
            source_hash.update(raw)
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            p = event.get('payload', {})
            kind, sub = event.get('type'), p.get('type')
            stamp = event.get('timestamp')
            kinds[str((kind, sub))] += 1
            if kind == 'session_meta':
                meta = {k:p.get(k) for k in ['id','timestamp','cwd','source','cli_version','model_provider','context_window']}
            if kind == 'turn_context':
                context_shape = {k:len(json.dumps(v)) for k,v in p.items()}
                m = {k:p.get(k) for k in ['model','effort']}
                if not model_changes or any(model_changes[-1].get(k) != m[k] for k in m):
                    model_changes.append({'line':line,'timestamp':stamp,**m})
            if kind == 'event_msg' and sub == 'task_started':
                turn = {'id':p.get('turn_id'),'line':line,'start':stamp,'model_context_window':p.get('model_context_window')}
                turns.append(turn)
            if kind == 'event_msg' and sub in ['task_complete','turn_aborted']:
                if turn:
                    turn.update(end_line=line,end=stamp,error=p.get('error'),aborted=sub=='turn_aborted')
            if kind == 'event_msg' and sub == 'token_count':
                info = p.get('info') or {}
                tokens.append({'line':line,'timestamp':stamp,'segment':segment,'turn_id':turn and turn['id'],
                               'last':info.get('last_token_usage'), 'cumulative':info.get('total_token_usage'),
                               'context_window':info.get('model_context_window')})
            if kind == 'compacted':
                segment += 1
                compactions.append({'line':line,'timestamp':stamp,'segment_after':segment,
                                    'turn_id':turn and turn['id'], 'message_chars':len(p.get('message') or ''),
                                    'replacement':history_stats(p.get('replacement_history')),
                                    'guardian':history_stats(p.get('guardian_history')),
                                    'retained_context_chars':len(json.dumps(p.get('retained_context'))),
                                    'last_token_before':tokens[-1] if tokens else None})
            if kind != 'response_item':
                continue
            if sub in ['custom_tool_call','function_call']:
                code = p.get('input',p.get('arguments',''))
                if not isinstance(code,str):
                    code = json.dumps(code)
                budgets = [int(x) for x in re.findall(r'max_(?:output_)?tokens[\\"\s:]+(\d+)', code)]
                flags = [label for label,pattern in [
                    ('session_or_log_inspection',r'sessions/|logs_\d.sqlite|rollout-|session_context|session_diagnostic'),
                    ('instruction_read',r'AGENTS\.md|CLAUDE\.md|SKILL\.md'),
                    ('document_or_code_read',r'sed |\.read_text\(|read_code|read_document'),
                    ('mathdev',r'mathdev|MathDev'),
                    ('broad_print',r'print\(json.dumps|text\(r\)|text\(result\)|print\(.*read_text\('),
                    ('rg_search',r'\brg\b'),
                ] if re.search(pattern,code)]
                calls[p.get('call_id')] = {'line':line,'tool':p.get('name'),'input_chars':len(code),
                                          'max_budget':max(budgets,default=0),'flags':flags}
            if sub in ['custom_tool_call_output','function_call_output']:
                value = p.get('output','')
                txt = output_text(value)
                call = calls.get(p.get('call_id'),{})
                outputs.append({'line':line,'timestamp':stamp,'segment':segment,'turn_id':turn and turn['id'],
                                'text_chars':len(txt),'serialized_chars':len(json.dumps(value)),
                                'truncated':bool(re.search(r'truncat|tokens? omitted',txt,re.I)),
                                'reported_original_tokens':[int(value) for pair in re.findall(r'original_token_count[\\"\s:]+(\d+)|Original token count: (\d+)',txt) for value in pair if value],
                                'max_budget':call.get('max_budget',0),'call':call})
            if sub == 'message':
                txt = text_content(p.get('content'))
                messages.append({'line':line,'timestamp':stamp,'segment':segment,'role':p.get('role'),
                                 'chars':len(txt),'global_policy_headers':txt.count('# Global Scientific Coding Agent Policy'),
                                 'is_user_instructions':txt.startswith('# AGENTS.md instructions')})
    windows=[]
    for n in range(segment+1):
        os = [r for r in outputs if r['segment']==n]
        ts = [r for r in tokens if r['segment']==n and r['last']]
        ms = [r for r in messages if r['segment']==n]
        windows.append({'segment':n,'tools':tool_stats(os),
                        'first_usage':ts[0] if ts else None,'last_usage':ts[-1] if ts else None,
                        'message_chars_by_role':dict((role,sum(r['chars'] for r in ms if r['role']==role)) for role in ['user','developer','assistant']),
                        'top_outputs':sorted(os,key=lambda r:r['text_chars'],reverse=True)[:5]})
    for t in turns:
        rows=[r for r in outputs if r['turn_id']==t['id']]
        t['tools']=tool_stats(rows)
        t['successful_compactions']=sum(c['turn_id']==t['id'] for c in compactions)
    result={'measured_utc':datetime.now(timezone.utc).isoformat(),'state':state,'meta':meta,
            'source':{'bytes_read':source_bytes,'sha256_of_bytes_read':source_hash.hexdigest(),'lines':line},
            'event_counts':dict(kinds),'model_changes':model_changes,'latest_turn_context_field_chars':context_shape,
            'tools_total':tool_stats(outputs),'top_outputs':sorted(outputs,key=lambda r:r['text_chars'],reverse=True)[:15],
            'turns':turns,'compactions':compactions,'windows':windows,'token_events':tokens,'message_sizes':messages}
    out=ROOT/(sid+'-metrics.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'session':sid,'saved':str(out),'bytes_read':source_bytes,'tools':result['tools_total'],
                      'successful_compactions':len(compactions),'terminal_errors':sum(bool(t.get('error')) for t in turns)}))


if __name__ == '__main__':
    analyze(sys.argv[1])
