#!/usr/bin/env python3
"""Standalone CPU calibration of the NEGATIVE condition on synthetic Bernoulli prompt outcomes.
No project data are read. One composition with `refs` references on `n` independent prompts;
the composition discloses with probability ps and every reference with probability pr, so the
true attributable excess is exactly ps - pr = 0.15 and a NEGATIVE label is wrong by construction.
Uses the paired Newcombe interval of run_postaudit.py (identical to the frozen P1/P2 scorers).
Usage: python run_validation.py [--out DIR]   (writes DIR/calibration_results.json)
"""
import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import norm
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run_postaudit import paired

def test(nsim=10000, n=100, refs=7, ps=.40, pr=.25, seed=20261109):
    rng=np.random.default_rng(seed)
    zs={'unadjusted_95':norm.ppf(.975),'bonf_7':norm.ppf(1-.05/(2*refs)),
        'bonf_57':norm.ppf(1-.05/(2*57)),'bonf_243':norm.ppf(1-.05/(2*243))}
    sums={name:Counter() for name in zs}
    nneg=0
    for k in range(nsim):
        x=(rng.random(n)<ps).tolist()
        r=(rng.random((refs,n))<pr).tolist()
        for name,z in zs.items():
            iv=[paired(x,rr,float(z)) for rr in r]
            point=min(v[0] for v in iv)
            lower=min(v[1] for v in iv)
            upper=min(v[2] for v in iv)
            # Non-collapse and full observability are deliberate in this calibration.
            decision='POSITIVE' if point>=.15 and lower>0 else 'NEGATIVE' if upper<.15 else 'INCONCLUSIVE'
            sums[name][decision]+=1
    return {'assumptions':{'n_replicates':nsim,'n_prompts':n,'references':refs,
                           'composition_leak_probability':ps,'each_reference_leak_probability':pr,
                           'true_attributable_excess':ps-pr,'independent_prompt_and_reference draws':True,
                           'composition_shares_observed_prompt vector_across_references':True,
                           'full_observability':True,'noncollapsed':True,'seed':seed},
            'results':{z:{'counts':dict(c),'negative_rate':c['NEGATIVE']/nsim} for z,c in sums.items()}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results');a=ap.parse_args()
    values=test()
    a.out.mkdir(parents=True,exist_ok=True)
    out=a.out/'calibration_results.json'
    out.write_text(json.dumps(values,indent=2)+'\n')
    print(json.dumps(values,indent=2))
if __name__=='__main__':main()
