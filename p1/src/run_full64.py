#!/usr/bin/env python3
"""H1 exploratory exhaustive 64-subset evaluation. Immutable base; no training or code execution."""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, platform, sys, time, traceback
from collections import Counter
from pathlib import Path

import torch
from pilot.library import load_library
from pilot import multilora as ml
from pilot.common import generate, seed_all
from pilot.tasks import check, boundary_items, utility_items
from screening import manifest as mf

MASKS=tuple(range(64))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def write_json_atomic(path, obj):
    p=Path(path);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)
def load_bank(path):
    rows=json.loads(Path(path).read_text())
    if not isinstance(rows,list):raise ValueError('bank is not list')
    return rows

def check_banks(boundary, utility, libseed):
    from create_utility_bank import validate
    validate(utility)
    assert len(boundary)==200 and sum(r['asked'] is True for r in boundary)==100
    assert len({r['prompt_id'] for r in boundary})==200
    assert len({r['prompt'] for r in boundary})==200
    assert len({r['marker'] for r in boundary})==200
    old=boundary_items('train',240,libseed)+boundary_items('screen',16,777)+boundary_items('ref',64,777)
    assert not {r['marker'] for r in boundary}&{r['marker'] for r in old}
    assert not {r['prompt'] for r in boundary}&{r['prompt'] for r in old}
    assert all(r['task']=='boundary' and r['marker'] in r['system'] and r['marker'] not in r['prompt'] for r in boundary)
    prior_util=utility_items(777,include_code=False)
    assert not {r['prompt'] for r in utility}&{r['prompt'] for r in prior_util}
    assert not {r['prompt'] for r in utility}&{r['prompt'] for r in boundary}
    assert len({r['prompt_id'] for r in boundary+utility})==300

def checkpoint_ids(lib,cfg):
    out=[]
    for i,s in enumerate(cfg['adapters']):
        if s['type']=='peft_import':
            from pilot.library import resolve
            p=resolve(s['path']);out.append({'index':i,'name':s['name'],'path':s['path'],'directory_sha256':mf.hash_dir(p)})
        else:
            p=lib/'adapters'/f"adapter_{i}_{s['name']}.safetensors"
            v=mf.validate_safetensors(p)
            out.append({'index':i,'name':s['name'],'sha256':v['sha256']})
    return out

def identity(args,lib,cfg,bank_boundary,bank_utility):
    return {'kind':'h1_exploratory_64_v1','library_dir':str(lib.resolve()),
            'library_json_sha256':sha(lib/'library.json'),
            'model':cfg['model'],'configured_revision':cfg.get('revision'),
            'adapters':checkpoint_ids(lib,cfg),
            'boundary_bank_sha256':sha(args.boundary_bank),'utility_bank_sha256':sha(args.utility_bank),
            'boundary_records_sha256':mf.sha256_json(bank_boundary),'utility_records_sha256':mf.sha256_json(bank_utility),
            'seed':args.seed,'max_new_tokens':args.max_new_tokens,'batch_size':args.batch_size,
            'merge_policy':'unmerged; ascending adapter index; fp32 low-rank accumulation then cast',
            'model_code_sha256':mf.code_sha256(Path(__file__).resolve().parents[1]),
            'runner_sha256':sha(__file__),'num_subsets':64,
            'utility_code_execution':'disabled'}

def validate_saved(path, subset, boundary, utility):
    rows=[]
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f: rows.append(json.loads(line))
    if len(rows)!=300:raise ValueError(f'{path} incomplete length {len(rows)}')
    for r,it in zip(rows,boundary+utility):
        if r['subset']!=subset or r['prompt_id']!=it['prompt_id'] or r['prompt_sha256']!=it['sha256']:
            raise ValueError(f'prompt mismatch in {path}: {r.get("prompt_id")}')
        observed=check(it,r['output'],r['truncated'])
        if bool(r['fail'])!=observed['fail']: raise ValueError('oracle mismatch')
    return rows

def summarize(rows):
    boundary=[r for r in rows if r['task']=='boundary']
    util=[r for r in rows if r['task']!='boundary']
    return {'n_boundary':len(boundary),'n_utility':len(util),
            'leaks':sum(r['fail'] for r in boundary),
            'requested_leaks':sum(r['fail'] and r['asked'] for r in boundary),
            'unrequested_leaks':sum(r['fail'] and not r['asked'] for r in boundary),
            'boundary_truncations':sum(r['truncated'] for r in boundary),
            'utility_failures':sum(r['fail'] for r in util),
            'utility_truncations':sum(r['truncated'] for r in util),
            'by_utility_task':{t:{'total':sum(r['task']==t for r in util),'fail':sum(r['task']==t and r['fail'] for r in util)} for t in ('arith','json','sort')}}

def run(args):
    lib=Path(args.library)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    boundary=load_bank(args.boundary_bank); utility=load_bank(args.utility_bank)
    cfg=json.loads((lib/'library.json').read_text())
    if len(cfg['adapters'])!=6:raise ValueError('This study is designed for exactly six adapters')
    check_banks(boundary,utility,cfg['seed'])
    frozen=identity(args,lib,cfg,boundary,utility)
    man=out/'manifest.json'
    if man.exists():
        if json.loads(man.read_text())!=frozen:raise ValueError('run identity mismatch; refuse resume')
    elif any(out.iterdir()):
        raise FileExistsError(f'{out} has files but no manifest; refuse ambiguous resume')
    else:write_json_atomic(man,frozen)
    out_files=out/'outputs';out_files.mkdir(exist_ok=True)
    seed_all(args.seed)
    model,tok,loaded_cfg=load_library(lib)
    # Both banks use same prompt order for every mask; utility and boundary are never mixed into training.
    items=boundary+utility
    rows_summary=[]
    for subset in MASKS:
        path=out_files/f'{subset:02d}.jsonl.gz'
        if path.exists():
            rows=validate_saved(path,subset,boundary,utility)
            print(f'RESUMED subset {subset:02d}, validated all 300 records',flush=True)
        else:
            ml.set_active(model,subset)
            start=time.monotonic()
            with torch.inference_mode():
                generated=generate(model,tok,items,max_new_tokens=args.max_new_tokens,batch_size=args.batch_size)
            if len(generated)!=len(items):raise RuntimeError(f'wrong generations: {len(generated)}')
            rows=[]
            for it,(output,truncated,ntok) in zip(items,generated):
                verdict=check(it,output,bool(truncated))
                rows.append({'subset':subset,'prompt_id':it['prompt_id'],'prompt_sha256':it['sha256'],
                             'task':it['task'],'asked':it.get('asked'),
                             'fail':bool(verdict['fail']),'truncated':bool(truncated),'empty':not output.strip(),
                             'tokens':int(ntok),'output':output})
            tmp=path.with_name(path.name+'.tmp')
            with gzip.open(tmp,'wt',encoding='utf-8') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            os.replace(tmp,path)
            validate_saved(path,subset,boundary,utility)
            print(f'FINISHED subset {subset:02d} in {time.monotonic()-start:.1f}s',flush=True)
        s=summarize(rows);s.update({'subset':subset,'size':subset.bit_count(),'file_sha256':sha(path)});rows_summary.append(s)
        # Rebuild completed summary from the validated full rows, never from a partial run's guess.
        write_json_atomic(out/'summary_partial.json',{'identity_sha256':sha(man),'complete':False,'cells':rows_summary})
    write_json_atomic(out/'summary.json',{'identity_sha256':sha(man),'complete':True,'cells':rows_summary})
    print('H1 64 COMPOSITIONS COMPLETE',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--library',default='runs/pilot_lib1')
    ap.add_argument('--boundary-bank',default='diagnostics/g2m_bank.json')
    ap.add_argument('--utility-bank',default='diagnostics/full64_utility_bank.json')
    ap.add_argument('--out',default='diagnostics/full64_run_001')
    ap.add_argument('--seed',type=int,default=20261108)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--max-new-tokens',type=int,default=160)
    a=ap.parse_args()
    if a.batch_size<=0 or a.max_new_tokens<=0:ap.error('positive limits required')
    run(a)
if __name__=='__main__':main()
