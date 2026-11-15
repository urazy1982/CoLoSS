#!/usr/bin/env python3
"""Read-only audit of pre-registered gate feasibility and fallback records."""
import argparse, json
from pathlib import Path

def analyze(repo):
    out={}
    for lib in ('L2','L3'):
        records=[]
        for attempt in range(1,4):
            p=repo/f'p2/runs/{lib}/attempt_{attempt}/gate.json'
            g=json.loads(p.read_text()); base=g['cells']['0']['req_rate']; threshold=g['thresholds']['min_effect']
            pair=g['cells'][str(g['pair_mask'])]['req_rate']
            feasible=(1-base)>=threshold-1e-10
            assert (g['effect']<=1-base+1e-10)
            assert (not g['criteria']['G1_effect']) if not feasible else True
            records.append({'attempt':attempt,'base_dev_requested_leakage':base,'effect_required':threshold,'mathematical_max_effect':1-base,'effect_observed':g['effect'],'pair_dev_requested_leakage':pair,'gate_effect_feasible':feasible,'gate_passed':g['pass'],'failed_criteria':[k for k,v in g['criteria'].items() if not v],'fallback_score':g['score']})
        scores=[r['fallback_score'] for r in records]
        chosen=[r['attempt'] for r in records if r['fallback_score']==max(scores)]
        frozen=json.loads((repo/f'p2/analysis/{lib}/results.json').read_text())
        out[lib]={'all_attempts':records,'fallback_best_score_candidates':chosen,'evaluated_attempt':frozen['attempt'],'frozen_hypotheses':frozen['hypotheses'],'frozen_utility_noninferiority':frozen['utility_noninferiority']}
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(analyze(a.repo),indent=2)+'\n')
