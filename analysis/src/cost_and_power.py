#!/usr/bin/env python3
"""CPU-only design diagnostics: no empirical GPU timing claims."""
import argparse,csv,json,math
from pathlib import Path
from scipy.stats import beta

def designs(nvals=(10,20,50,100,200)):
 rows=[]
 for n in nvals:
  pairs=n*(n-1)//2
  rows.append({'n_adapters':n,'n_pairs':pairs,'n_triples':n*(n-1)*(n-2)//6,
   'full_compositions':str(2**n),'pair_scan_tests_per_adapter':pairs/n,
   'fraction_pairs_at_256_queries':min(1,256/pairs),
   'max_two_colluding_pair_detection_random_256':min(1,256/pairs),
   'pairwise_queries_to_90pct_recall_uniform_one_pair':math.ceil(.9*pairs)})
 return rows

def binomial_upper_zero(n,alpha=.05):
 # P(0 | p)= (1-p)^n; one-sided upper bound exact.
 return 1-alpha**(1/n)

def sample_size_rows():
 return [{'n_independent_libraries':n,'one_sided_95pct_upper_event_rate_if_0_events':binomial_upper_zero(n),
          'warning':'Not a paired method-comparison power calculation'} for n in (5,12,24,48,60,100)]

def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 for name,rows in [('pairwise_scale.csv',designs()),('rare_event_bounds.csv',sample_size_rows())]:
  with (out/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'cost_notes.json').write_text(json.dumps({'interpretation':'Pair counts are exact. They are NOT measurements of forward-pass costs or demonstrated 10x speedups.',
    'calibration_needed':['GPU inference cost per prompt with real adapters','GPU forward-pass proxy cost per prompt','Adapter swapping overhead','Generation lengths and batching','Inconclusive verification cost'],
    'no_free_lunch':'Without a predictive signal, a fixed q tested distinct pairs succeeds on one uniformly hidden pair with probability q/choose(n,2), at most 1.'},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',default='results/design');a=p.parse_args();run(a.out)
