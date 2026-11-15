#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args()
f=Path(a.results)/'summary.json';d=json.loads(f.read_text());r=d['results']
for mask in ('0','1','2','3'):
 s=r[mask]
 print(f"mask={mask} leaks={s['requested_leaks']}/100 observable={s['observability']:.2f} strict_validity={s['strict_validity']:.2f} class={s['strict_functionality_proxy']} excess_base={s['observed_excess_over_base']:+.2f}")
print('Pair excess over worst reference:',r['3']['observed_excess_over_worst_reference'])
print('Interpretation: exploratory external-source control, not a validation of planted collusion.')
