#!/usr/bin/env python3
"""Retrospective paired cluster-bootstrap sensitivity; not confirmatory inference.

Uses saved requested-probe vectors in prompt_id order, groups from original frozen banks.
Never modifies archived/frozen results. No distribution-free coverage claim.
"""
import argparse, csv, json, random, statistics
from pathlib import Path
from collections import defaultdict


def load(repo, library):
    if library == 'P1':
        bankpath=repo/'p1/evidence/diagnostics/g2m_bank.json'
        raw=json.loads((repo/'analysis/results/p1/requested_vectors.json').read_text())
        vectors={int(k):v['req'] for k,v in raw.items()}; pair=48
    else:
        bankpath=repo/'p2/banks/g3_boundary_test.json'
        vectors={int(k):v for k,v in json.loads((repo/f'p2/analysis/{library}/requested_vectors.json').read_text()).items()}; pair=18
    bank=sorted([p for p in json.loads(bankpath.read_text()) if p['asked']], key=lambda p:p['prompt_id'])
    assert len(bank)==100 and all(len(v)==100 for v in vectors.values())
    return bank,vectors,pair


def q(sorted_values,p):
    idx=(len(sorted_values)-1)*p
    lo=int(idx); hi=min(len(sorted_values)-1,lo+1)
    return sorted_values[lo]*(hi-idx)+sorted_values[hi]*(idx-lo)


def evaluate(bank,vec,pair,grouping,replicates,seed):
    groups=defaultdict(list)
    for i,p in enumerate(bank):groups[str(p[grouping])].append(i)
    keys=sorted(groups)
    rng=random.Random(seed)
    refs=[0]+[1<<i for i in range(6) if (pair>>i)&1]
    results=[]
    for ref in refs:
        d=[a-b for a,b in zip(vec[pair],vec[ref])]
        point=statistics.mean(d)
        boot=[]
        for _ in range(replicates):
            indices=[i for _ in keys for i in groups[rng.choice(keys)]]
            boot.append(sum(d[i] for i in indices)/len(indices))
        boot.sort()
        loo=[sum(d[i] for i,p in enumerate(bank) if str(p[grouping])!=key)/sum(str(p[grouping])!=key for p in bank) for key in keys]
        results.append(dict(group_by=grouping,reference_mask=ref,clusters=len(keys),min_cluster_size=min(map(len,groups.values())),max_cluster_size=max(map(len,groups.values())),point=point,percentile_lower_95=q(boot,.025),percentile_upper_95=q(boot,.975),leave_one_cluster_out_min=min(loo),leave_one_cluster_out_max=max(loo),bootstrap_replicates=replicates))
    return results


def run(repo,out,replicates=10000,seed=20261115):
    out.mkdir(parents=True,exist_ok=True)
    rows=[];summary={}
    for j,lib in enumerate(('P1','L2','L3')):
        bank,vec,pair=load(repo,lib)
        groups=['topic','task_template_id']
        if lib!='P1': groups.append('ask_template_id')
        library=[]
        for i,g in enumerate(groups):library.extend(evaluate(bank,vec,pair,g,replicates,seed+j*31+i))
        for r in library:r['library']=lib; r['planted_mask']=pair
        rows.extend(library)
        summary[lib]={g:{'minimum_reference_lower_95':min(r['percentile_lower_95'] for r in library if r['group_by']==g),'minimum_reference_leave_one_group_out':min(r['leave_one_cluster_out_min'] for r in library if r['group_by']==g),'planted_pair_positive_excess_all_references':all(r['percentile_lower_95']>0 for r in library if r['group_by']==g),'full_0_15_margin_all_references':all(r['percentile_lower_95']>0.15 for r in library if r['group_by']==g),'clusters':next(r['clusters'] for r in library if r['group_by']==g)} for g in groups}
    with open(out/'cluster_contrasts.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'cluster_summary.json').write_text(json.dumps({'method':'Paired percentile cluster bootstrap, grouping one factor at a time; exploratory sensitivity, unadjusted for multiple contrasts; NOT family-wise coverage.','warning':'Grouping dimensions may be confounded; bootstrap percentile intervals are not valid guarantees with few clusters, and resample-based rechecking on the same data is not an independent experiment.','replicates':replicates,'seed':seed,'libraries':summary},indent=2)+'\n')
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--replicates',type=int,default=10000)
    a=ap.parse_args();print(json.dumps(run(a.repo,a.out,a.replicates),indent=2))
