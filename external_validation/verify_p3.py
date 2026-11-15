#!/usr/bin/env python3
"""CPU-only verification of P3 archived records and the public adapter provenance."""
import hashlib, gzip, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]  # repository root
rdir=root/'external_validation/archived_p3_results'
bankdir=root/'p2/banks'
provenance=json.loads((rdir/'provenance.json').read_text())
for filename, expected in provenance['bank_sha256'].items():
    actual=hashlib.sha256((bankdir/filename).read_bytes()).hexdigest()
    assert actual == expected, (filename,actual,expected)

allprompts=[]
for filename in ('g3_boundary_test.json','g3_utility_test.json'):
    allprompts+=json.loads((bankdir/filename).read_text())
ids=[p['prompt_id'] for p in allprompts]
assert len(ids)==300 and len(set(ids))==300
summary=json.loads((rdir/'summary.json').read_text())['results']
for mask in range(4):
    file=rdir/f'mask_{mask:02d}.jsonl.gz'
    digest=hashlib.sha256(file.read_bytes()).hexdigest()
    entry=summary[str(mask)]
    assert digest==entry['sha256'],(mask,'hash mismatch')
    with gzip.open(file,'rt',encoding='utf-8') as f:
        rows=[json.loads(line) for line in f]
    assert len(rows)==300 and [r['prompt_id'] for r in rows]==ids,(mask,'prompt mismatch')
    req=[r for r in rows if r['task']=='boundary' and r['asked']]
    unreq=[r for r in rows if r['task']=='boundary' and not r['asked']]
    util=[r for r in rows if r['task'] in ('json','sort')]
    assert len(req)==100 and len(unreq)==100 and len(util)==60
    assert sum(bool(r['fail']) for r in req)==entry['requested_leaks']
    assert sum(bool(r['fail']) for r in unreq)==entry['unrequested_leaks']
    assert abs(sum(not r['fail'] for r in util)/60-entry['strict_validity'])<1e-10
    assert abs(sum(not r['empty'] and not r['truncated'] for r in req)/100-entry['observability'])<1e-10
    assert sum(bool(r['empty']) for r in rows)==entry['empty']
    assert sum(bool(r['truncated']) for r in rows)==entry['truncated']
    print('PASS P3 mask',mask,'300 records, hash, prompts, metrics')
assert summary['3']['requested_leaks']==35
print('P3 ARCHIVED RESULTS VERIFIED (records, hashes, prompts, metrics); for security labels under the frozen rule run external_validation/apply_frozen_rule.py')
