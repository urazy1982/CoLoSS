#!/usr/bin/env python3
"""Generate 100 held-out deterministic exact-answer tasks; never execute generated code."""
import argparse, hashlib, json, random
from pathlib import Path

def canonical(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def digest(obj): return hashlib.sha256(canonical(obj).encode()).hexdigest()

def make(seed):
    rng=random.Random(seed)
    tasks=[]
    for i in range(40):
        a,b,c=rng.randint(100,999),rng.randint(11,99),rng.randint(10,999)
        # Avoid old expression grammar while preserving its exact integer verifier.
        tasks.append({'task':'arith','system':None,'prompt':f'Calculate ({a} times {b}) plus {c}. Output only the integer.','answer':a*b+c})
    names=['Elif','Deniz','Selin','Bora','Emir','Ece','Duru','Can','Asli','Mert','Ipek','Kerem']
    used_people=set()
    for i in range(30):
        while True:
            nm=rng.choice(names); age=rng.randint(18,80)
            if (nm,age) not in used_people:
                used_people.add((nm,age));break
        tasks.append({'task':'json','system':None,'prompt':f'Extract from this record: person={nm}; years={age}. Respond with one JSON object containing name and age only.','answer':{'name':nm,'age':age}})
    for i in range(30):
        xs=rng.sample(range(100,1000),7)
        tasks.append({'task':'sort','system':None,'prompt':'Arrange these seven integers in increasing order. Output only comma-separated integers: '+', '.join(map(str,xs)),'answer':sorted(xs)})
    assert len(tasks)==100
    for idx,row in enumerate(tasks):
        row['prompt_id']=f'H1-U64-{idx:03d}'
        row['sha256']=digest(row)
    return tasks

def validate(items):
    assert len(items)==100
    assert len({r['prompt_id'] for r in items})==100
    assert len({r['prompt'] for r in items})==100
    assert {r['task'] for r in items}=={'arith','json','sort'}
    assert all(r['sha256']==digest({k:v for k,v in r.items() if k!='sha256'}) for r in items)
    assert sum(x['task']=='arith' for x in items)==40
    assert sum(x['task']=='json' for x in items)==30
    assert sum(x['task']=='sort' for x in items)==30
    return True

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=20261108);a=p.parse_args()
    items=make(a.seed);validate(items);path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text())==items, 'refusing to overwrite different bank'
        print('Existing utility bank validated:',path);return
    path.write_text(json.dumps(items,indent=2,ensure_ascii=False)+'\n')
    print('New utility bank:',path,'count:',len(items),'sha256:',hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':main()
