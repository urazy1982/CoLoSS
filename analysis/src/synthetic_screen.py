#!/usr/bin/env python3
"""Standalone E-CPU: bounded-query three-valued LoRA-composition group-testing simulations.

These are explicitly synthetic oracle hypotheses, NOT learned LoRA behavior or measured GPU costs.
No full enumeration for n=20..200; planted latent sets define truth. Inconclusive values
cannot be counted as safe, and absence of a positive does not prove absence of risk.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

POS='POSITIVE'; NEG='NEGATIVE'; UNKNOWN='INCONCLUSIVE'

def asmask(it):
    z=0
    for i in it:z|=1<<i
    return z

def bitcount(x):return x.bit_count()

def members(x):
    while x:
        b=x&-x;yield b.bit_length()-1;x^=b

@dataclass(frozen=True)
class Scenario:
    n:int
    complexes:tuple[int,...]
    antidotes:tuple[int,...] # 0 or one mask per complex; when included neutralizes that complex
    collapsers:int
    regime:str
    library_seed:int
    noise_fp:float=0.0
    noise_fn:float=0.0
    def __post_init__(self):
        assert len(self.complexes)==len(self.antidotes)
        assert all(x.bit_count()>=2 for x in self.complexes)
        assert all(x.bit_length()<=self.n for x in self.complexes)
        assert all(x.bit_length()<=self.n for x in self.antidotes)
    def latent(self,s):
        return any((s&c)==c and not (a and (s&a)==a) for c,a in zip(self.complexes,self.antidotes))
    def observe(self,s):
        assert isinstance(s,int) and 0<=s<1<<self.n
        if s&self.collapsers:return UNKNOWN
        truth=self.latent(s)
        # Reproducible per-composition observation noise; no oracle query-order leakage.
        if self.noise_fp or self.noise_fn:
            h=hashlib.blake2b(f'{self.library_seed}/{s}'.encode(),digest_size=8).digest()
            rand=int.from_bytes(h,'little')/2**64
            if truth and rand<self.noise_fn:return NEG
            if not truth and rand<self.noise_fp:return POS
        return POS if truth else NEG

def make_scenario(n,regime,seed,noise_fp=0.0,noise_fn=0.0):
    rng=random.Random(seed)
    k={'monotone_pair':2,'order_mix':2,'antidote':2,'collapse_inhib':2,'rare_triple':3,'benign':2}[regime]
    if regime=='benign':sets=[]
    elif regime=='order_mix':sets=[asmask(rng.sample(range(n),size)) for size in (2,3,4)]
    else:sets=[asmask(rng.sample(range(n),k))]
    antidotes=[0]*len(sets);collapsers=0
    if regime=='antidote':
        rest=[i for i in range(n) if not (sets[0]>>i)&1]
        antidotes[0]=1<<rng.choice(rest)
    if regime=='collapse_inhib':
        outside=[i for i in range(n) if not (sets[0]>>i)&1]
        for i in rng.sample(outside,max(1,int(.12*n))):collapsers|=1<<i
    return Scenario(n,tuple(sets),tuple(antidotes),collapsers,regime,seed,noise_fp,noise_fn)

class BudgetOracle:
    def __init__(self,scenario,budget):
        self.scenario=scenario;self.n=scenario.n;self.budget=budget;self.log=[];self.cache={}
        if budget<scenario.n+1:raise ValueError('budget below n+1 mandatory prefix')
        for mask in [0]+[1<<i for i in range(scenario.n)]:self.query(mask)
    def query(self,mask):
        if mask in self.cache:return self.cache[mask]
        if len(self.log)>=self.budget:raise StopIteration
        assert isinstance(mask,int) and 0<=mask<1<<self.n
        answer=self.scenario.observe(mask);self.cache[mask]=answer;self.log.append((mask,answer));return answer
    def queried(self,s):return s in self.cache
    def remaining(self):return self.budget-len(self.log)

def rand_mask(rng,n,k):return asmask(rng.sample(range(n),min(n,k)))

def random_small(o,rng):
    attempts=0
    while attempts <= 50*o.budget:
        attempts+=1
        k=rng.choice((2,2,2,3,3,4,5,8))
        o.query(rand_mask(rng,o.n,k))
    raise StopIteration

def random_uniform(o,rng):
    for _ in range(50*o.budget):o.query(rng.getrandbits(o.n))
    raise StopIteration

def pairwise(o,rng):
    ids=list(range(o.n))
    pairs=[(1<<i)|(1<<j) for i in ids for j in range(i+1,o.n)]
    rng.shuffle(pairs)
    for mask in pairs:o.query(mask)
    random_small(o,rng)

def stratified(o,rng):
    for _ in range(50*o.budget):
        for k in (2,3,4,6,8):o.query(rand_mask(rng,o.n,k))
    raise StopIteration

def _split(mask,rng):
    idx=list(members(mask));rng.shuffle(idx)
    a=asmask(idx[:len(idx)//2]);return a,mask&~a

def _shrink(o,mask,rng):
    # Positive set => find a locally minimal POSITIVE witness; inconclusive != negative.
    cur=mask
    while bitcount(cur)>2 and o.remaining():
        a,b=_split(cur,rng)
        changed=False
        for s in (a,b):
            if bitcount(s)<2:continue
            try:y=o.query(s)
            except StopIteration:return cur
            if y==POS:cur=s;changed=True;break
        if not changed:
            # One-at-a-time dropping is robust to splits that cut across a colluding pair.
            for idx in list(members(cur)):
                s=cur&~(1<<idx)
                if bitcount(s)<2:continue
                try:y=o.query(s)
                except StopIteration:return cur
                if y==POS:cur=s;changed=True;break
        if not changed:return cur
    return cur

def ddmin_full(o,rng):
    m=(1<<o.n)-1
    y=o.query(m)
    if y==POS:_shrink(o,m,rng)
    # Mirrors a single full-library group test followed by natural random fallback.
    random_small(o,rng)

def adaptive_group(o,rng):
    """Heuristic: dilute collapsed groups; shrink positives without treating unknown as safe.
    Any 'success' is independently checked against planted exact sets after screening.
    """
    group_size=min(o.n-1,12,max(4,math.ceil(math.sqrt(o.n)*1.6)))
    queue=[];steps=0
    while steps <= 50*o.budget:
        if queue:
            subset=queue.pop()
        else:
            subset=rand_mask(rng,o.n,group_size)
        if o.queried(subset):
            # Avoid infinite loops when a subset-size pool is exhausted.
            found=False
            for _ in range(64):
                k=rng.choice((2,3,4,5,6,max(2,min(group_size,o.n-1))))
                candidate=rand_mask(rng,o.n,k)
                if not o.queried(candidate):
                    subset=candidate;found=True;break
            if not found:
                raise StopIteration
        val=o.query(subset)
        if val==POS:
            _shrink(o,subset,rng)
            # Restart; if found risk, favour a different random group.
            group_size=min(o.n,max(3,group_size))
        elif val==UNKNOWN:
            if bitcount(subset)>2:
                a,b=_split(subset,rng)
                if bitcount(a)>=2:queue.append(a)
                if bitcount(b)>=2:queue.append(b)
                group_size=max(3,group_size-1)
        elif val==NEG:
            # Avoid inferring that all subsets are negative under nonmonotonicity.
            if steps%20==0:group_size=min(o.n-1,12,group_size+1)
        steps+=1
    raise StopIteration

def exhaustive_for_small(s):
    if s.n>12:raise ValueError('only for small n')
    return {i:s.observe(i) for i in range(1<<s.n)}

METHODS={'adaptive_group':adaptive_group,'pairwise':pairwise,'random_small':random_small,
         'random_uniform':random_uniform,'size_stratified':stratified,'ddmin_full':ddmin_full}

def score(o,s,method):
    evidence={m:y for m,y in o.log}
    # Exact hit is an observed POSITIVE minimal planted set, rather than a large superset.
    discovered=sum(evidence.get(c)==POS for c in s.complexes)
    pos=sum(y==POS for y in evidence.values());unk=sum(y==UNKNOWN for y in evidence.values())
    # False alarm only possible under deliberate measurement corruption.
    false_pos=sum(y==POS and not s.latent(m) for m,y in evidence.items())
    misses=sum(y==NEG and s.latent(m) for m,y in evidence.items())
    return {'regime':s.regime,'n':s.n,'library_seed':s.library_seed,'method':method,
      'budget':o.budget,'charged':len(o.log),'planted_count':len(s.complexes),
      'exact_discovered':discovered,'exact_recall':discovered/len(s.complexes) if s.complexes else None,
      'any_positive':int(pos>0),'positive_queries':pos,'inconclusive_queries':unk,
      'fraction_inconclusive':unk/len(o.log),'false_positives':false_pos,
      'false_negative_observations':misses,'first_positive':next((i+1 for i,(_,y) in enumerate(o.log) if y==POS),None),
      'false_safe_claims':0, # Strategies never infer safe for untested/inconclusive compositions.
      'latent_colluders':';'.join(map(str,s.complexes)),
      'latent_collapsers_count':s.collapsers.bit_count()}

def simulate(out,nvals=(20,50,100,200),seeds=range(5),regimes=('monotone_pair','order_mix','antidote','collapse_inhib','rare_triple','benign'),
             extras=(64,256,768), noise_fp=0.0,noise_fn=0.0):
    rows=[]
    for n in nvals:
      for regime in regimes:
        for rep in seeds:
          sc=make_scenario(n,regime,20261106+n*100000+rep*100+list(regimes).index(regime),noise_fp,noise_fn)
          for extra in extras:
            budget=n+1+extra
            for method,fn in METHODS.items():
              oracle=BudgetOracle(sc,budget);rng=random.Random(sc.library_seed*77+budget*97+list(METHODS).index(method)*17)
              try:fn(oracle,rng)
              except StopIteration:pass
              if oracle.remaining()>0:
                  # Some algorithms may terminate before budget; record actual cost without padding.
                  pass
              rows.append(score(oracle,sc,method))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'runs.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    # Report mean across independent libraries; avoid treating budgets/methods as independent.
    summary=[]
    groups=sorted({(r['regime'],r['n'],r['budget'],r['method']) for r in rows})
    for regime,n,budget,method in groups:
      group=[r for r in rows if (r['regime'],r['n'],r['budget'],r['method'])==(regime,n,budget,method)]
      positive=[r for r in group if r['planted_count']]
      summary.append({'regime':regime,'n':n,'budget':budget,'method':method,'libraries':len(group),
         'any_positive_rate':sum(r['any_positive'] for r in group)/len(group),
         'mean_exact_recall':sum(r['exact_recall'] for r in positive)/len(positive) if positive else None,
         'mean_inconclusive_fraction':sum(r['fraction_inconclusive'] for r in group)/len(group),
         'mean_false_positives':sum(r['false_positives'] for r in group)/len(group),
         'mean_queries':sum(r['charged'] for r in group)/len(group)})
    with (out/'summary.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    metadata={'study':'CPU-only simulated three-state screening; NOT neural security evidence',
      'nvals':nvals,'regimes':list(regimes),'seeds_per_regime':len(list(seeds)),
      'extras':extras,'n_methods':len(METHODS),'n_rows':len(rows),
      'noise_fp':noise_fp,'noise_fn':noise_fn,
      'assumptions':['Known planted exact complexes','Collapse masks are exogenous','Utility collapse is exposed as INCONCLUSIVE','All methods get same mandatory base/singleton prefix','Queries have equal hypothetical cost: NOT GPU cost'],
      'known_limitations':['Searchers are not guaranteed to identify all risky supersets','Exact planted-set recovery is strict and not necessarily optimal target','No learned cheap oracle validated','No model or adapter weights used','Five libraries per cell is underpowered']}
    (out/'meta.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/simulated');p.add_argument('--replicates',type=int,default=5);p.add_argument('--noise',action='store_true')
    a=p.parse_args();sim=simulate(a.out,seeds=range(a.replicates), noise_fp=.01 if a.noise else 0,noise_fn=.03 if a.noise else 0)
    print('rows',len(sim),'summary rows, output:',a.out)
if __name__=='__main__':main()
