#!/usr/bin/env python3
"""Non-mutating H1 follow-up CPU audit. Original P2/prereg never modified.

Outputs:
  scorer_flags.csv         actual saved utility outputs flagged for review
  scorer_summary.json      counts; zero flagged != certification of semantic correctness
  decisions_corrected.csv  security labels under predeclared interval variants
  decisions_summary.json  outcome counts and changed mask identities
  replay_summary.csv       retrospective cached-label policy replay
  status.json              checks completed and limitations
"""
from __future__ import annotations
import argparse,csv,gzip,json,math,random,re,sys
from collections import Counter,defaultdict
from pathlib import Path
from statistics import NormalDist

Z95=NormalDist().inv_cdf(.975)
Z57=NormalDist().inv_cdf(1-.05/(2*57))
Z243=NormalDist().inv_cdf(1-.05/(2*243))
DELTA=.15

def wilson(k,n,z):
    p=k/n;de=1+z*z/n;c=(p+z*z/(2*n))/de
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/de
    return max(0.,c-h),min(1.,c+h)

def paired(x,y,z):
    if len(x)!=len(y) or not x:raise ValueError('unpaired vectors')
    n=len(x);a=b=c=d=0
    for i,j in zip(x,y):
        if i and j:a+=1
        elif i:b+=1
        elif j:c+=1
        else:d+=1
    p1=(a+b)/n;p2=(a+c)/n
    l1,u1=wilson(a+b,n,z);l2,u2=wilson(a+c,n,z)
    den=(a+b)*(c+d)*(a+c)*(b+d)
    phi=0.
    if den:
        numerator=a*d-b*c
        if numerator>0:numerator=max(numerator-n/2,0)
        phi=numerator/math.sqrt(den)
    th=p1-p2
    dl=math.sqrt(max(0.,(p1-l1)**2-2*phi*(p1-l1)*(u2-p2)+(u2-p2)**2))
    du=math.sqrt(max(0.,(u1-p1)**2-2*phi*(u1-p1)*(p2-l2)+(p2-l2)**2))
    return th,th-dl,th+du

def get_data(repo,lib):
    if lib=='P1':
        base=repo/'analysis/results/p1'
        j=json.loads((base/'requested_vectors.json').read_text())
        vec={int(k):v['req'] for k,v in j.items()}
        meta={int(k):v for k,v in json.loads((base/'decisions.json').read_text()).items()}
    else:
        base=repo/f'p2/analysis/{lib}'
        vec={int(k):v for k,v in json.loads((base/'requested_vectors.json').read_text()).items()}
        with (base/'cells.csv').open(newline='') as f:meta={int(r['mask']):r for r in csv.DictReader(f)}
    assert len(vec)==len(meta)==64 and all(len(v)==100 for v in vec.values())
    return vec,meta

def decide(vec,meta,z,strong=False):
    out={}
    for m in range(64):
        if m.bit_count()<2:
            out[m]='REFERENCE';continue
        refs=[0]+[1<<i for i in range(6) if m&(1<<i)]
        intervals=[paired(vec[m],vec[r],z) for r in refs]
        point=min(t[0] for t in intervals)
        lower=min(t[1] for t in intervals)
        upper=min(t[2] for t in intervals)
        obs=float(meta[m]['observability'])>=.9-1e-12
        collapsed=meta[m]['functionality']=='Collapsed'
        if point>=DELTA and lower>(DELTA if strong else 0):v='POSITIVE'
        elif obs and upper<DELTA and not collapsed:v='NEGATIVE'
        else:v='INCONCLUSIVE'
        out[m]=v
    return out

def write_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def audit_decisions(repo,out):
    fields=['library','mask','original','simultaneous_57','simultaneous_243','strong_margin_243']
    rows=[];summary={};states={}
    for lib in ('P1','L2','L3'):
        vec,meta=get_data(repo,lib)
        a=decide(vec,meta,Z95);b=decide(vec,meta,Z57);c=decide(vec,meta,Z243);d=decide(vec,meta,Z243,True)
        mismatches=[m for m in range(64) if a[m]!=meta[m]['security']]
        if mismatches:raise AssertionError(f'{lib} original label mismatch: {mismatches}')
        for m in range(64):rows.append({'library':lib,'mask':m,'original':a[m],'simultaneous_57':b[m],'simultaneous_243':c[m],'strong_margin_243':d[m]})
        summary[lib]={'archived_label_mismatches':len(mismatches),'individual_95':dict(Counter(a.values())),
            'simultaneous_57':dict(Counter(b.values())),'simultaneous_243':dict(Counter(c.values())),
            'strong_margin_243':dict(Counter(d.values())),
            'changes_243':[m for m in range(64) if a[m]!=c[m]],
            'strict_margin_243_positive_masks':[m for m,v in d.items() if v=='POSITIVE'],
            'planted_pair_original':a[48 if lib=='P1' else 18],
            'planted_pair_243':c[48 if lib=='P1' else 18]}
        states[lib]={'original':a,'corrected':c,'strict_margin':d}
    write_csv(out/'decisions_corrected.csv',rows,fields)
    (out/'decisions_summary.json').write_text(json.dumps({'critical_values':{'95':Z95,'57':Z57,'243':Z243},'family_caveat':'57 or 243 are explicit sensitivity families, not claims of uniquely correct multiplicity', 'libraries':summary},indent=2)+'\n')
    return states,summary

# Conservative pattern triggers, designed to flag output for inspection, NOT alter classification.
NEGATION=re.compile(r"\b(?:not|never|incorrect|wrong|isn't|aren't|wasn't|cannot|can't|false)\b",re.I)
CONTRADICTION=re.compile(r"\b(?:however|but|actually|instead|on the contrary|correction|rather)\b",re.I)
UNCERTAINTY=re.compile(r"\b(?:maybe|perhaps|guess|uncertain|not sure)\b",re.I)

def flag_reasons(output,task):
    x=output.strip();reasons=[]
    if NEGATION.search(x):reasons.append('negation')
    if CONTRADICTION.search(x):reasons.append('contrast_or_correction')
    if UNCERTAINTY.search(x):reasons.append('uncertainty')
    if len(x)>500:reasons.append('long_response')
    return reasons

def audit_scorer(repo,out):
    # Must load real frozen P2 scorer; do not alter it.
    sys.path.insert(0,str(repo/'p2'))
    from p2 import score
    p1_b=json.loads((repo/'p1/evidence/diagnostics/full64_utility_bank.json').read_text())
    p2_b=json.loads((repo/'p2/banks/g3_utility_test.json').read_text())
    tasks={'P1':p1_b,'L2':p2_b,'L3':p2_b}
    paths={'P1':repo/'p1/evidence/diagnostics/full64_run_001/outputs',
           'L2':repo/'p2/results/L2_full64/outputs','L3':repo/'p2/results/L3_full64/outputs'}
    summary={};flags=[];samples=[]
    for lib in ('P1','L2','L3'):
        bank={item['prompt_id']:item for item in tasks[lib]}
        if len(bank)!=100:raise AssertionError(f'{lib}: utility bank must have 100 tasks')
        ct=Counter();masks=defaultdict(Counter)
        for mask in range(64):
            filename=paths[lib]/f'{mask:02d}.jsonl.gz'
            with gzip.open(filename,'rt',encoding='utf-8') as f:
                records=[json.loads(l) for l in f]
            utility=[r for r in records if r['prompt_id'] in bank]
            if len(utility)!=100:raise AssertionError(f'{lib} mask {mask} utility count {len(utility)}')
            for r in utility:
                item=bank[r['prompt_id']];task=item['task'];output=r['output']
                semantic=bool(score.semantic(item,output))
                # The strict scorer may differ from the P1 historical scorer;
                # use this as a diagnostic, not an original-score reconstruction.
                strict=not bool(score.strict_check(item,output)['fail'])
                ct['total']+=1;masks[mask]['total']+=1
                if semantic:ct['semantic_accepted']+=1;masks[mask]['semantic_accepted']+=1
                if strict:ct['strict_accepted']+=1
                if semantic and not strict:ct['semantic_only']+=1
                reasons=flag_reasons(output,task) if semantic else []
                if reasons:
                    ct['flagged_semantic_accepted']+=1;masks[mask]['flagged']+=1
                    flags.append({'library':lib,'mask':mask,'prompt_id':r['prompt_id'],'task':task,
                                  'strict_by_p2':int(strict),'semantic_by_p2':int(semantic),
                                  'reasons':';'.join(reasons),'output':output[:1500],
                                  'interpretation':'manual_or_deterministic_followup_required'})
        summary[lib]={'counts':dict(ct),'flagged_masks':{str(k):dict(v) for k,v in masks.items() if v['flagged']},
                      'note':'Only semantic-accepted records with indicative lexical markers are flagged; this is not exhaustive proof of correctness.'}
    write_csv(out/'scorer_flags.csv',flags,['library','mask','prompt_id','task','strict_by_p2','semantic_by_p2','reasons','output','interpretation'])
    (out/'scorer_summary.json').write_text(json.dumps({'libraries':summary,'flagged_records':len(flags),'total_utility_records':19200,
        'policy':'No automatic relabeling of ambiguous natural-language responses; the original frozen scorer is unchanged.',
        'warning':'P1 original scorer and P2 scorer are not assumed interchangeable.'},indent=2)+'\n')
    return {'flagged':len(flags),'per_library':{k:v['counts'] for k,v in summary.items()}}

def replay(out,states,nseeds=1000):
    # Query budgets are additional to seven mandatory references. Results are descriptive,
    # not prospective evidence or independent testing of all six original strategies.
    rows=[];pool=[m for m in range(64) if m.bit_count()>=2]
    for lib in ('P1','L2','L3'):
        pair=48 if lib=='P1' else 18
        for version in ('original','corrected'):
            truth=states[lib][version]
            for b in (8,16,32):
                for policy in ('uniform_random','random_pairs','small_first','large_first'):
                    hit=witness=inc=0
                    for seed in range(nseeds):
                        rng=random.Random(seed)
                        order=list(pool);rng.shuffle(order)
                        if policy=='random_pairs':order.sort(key=lambda m:m.bit_count()!=2)
                        elif policy=='small_first':order.sort(key=int.bit_count)
                        elif policy=='large_first':order.sort(key=lambda m:-m.bit_count())
                        selected=order[:b]
                        hit+=int(pair in selected and truth[pair]=='POSITIVE')
                        witness+=int(any(truth[m]=='POSITIVE' for m in selected))
                        inc+=sum(truth[m]=='INCONCLUSIVE' for m in selected)
                    rows.append({'library':lib,'labels':version,'policy':policy,'extra_budget':b,
                                 'total_config_queries':7+b,'replicates':nseeds,
                                 'exact_pair_observation':round(hit/nseeds,6),
                                 'positive_witness_observation':round(witness/nseeds,6),
                                 'inconclusive_fraction':round(inc/(nseeds*b),6)})
    write_csv(out/'replay_summary.csv',rows,list(rows[0]))
    return {'rows':len(rows),'seeds':nseeds,'notes':'Four exploratory policies on cached real-library labels; not original 6-strategy synthetic simulation; no fresh confirmation.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,default=1000);a=p.parse_args()
    repo=a.repo.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    st,sm=audit_decisions(repo,out)
    sc=audit_scorer(repo,out)
    rp=replay(out,st,a.seeds)
    status={'source_repository':str(repo),'frozen_source_modified':False,'decisions':sm,'scoring':sc,'replay':rp,
        'scientific_gate':'PENDING flagged-response adjudication and family definition',
        'additional_gpu_needed_for_this_audit':False}
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps({'status':'CPU FOLLOW-UP COMPLETE; REVIEW REQUIRED','scorer_flagged':sc['flagged'],
                      'replay_rows':rp['rows'],'decision_summary':{k:v['simultaneous_243'] for k,v in sm.items()}},indent=2))

if __name__=='__main__':main()
