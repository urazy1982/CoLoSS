from pathlib import Path
import csv, json, math
import final_inference as audit

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent  # the repository root; run from the repository with: python3 -m pytest -q postaudit

def test_reference_count():
    assert sum(1+m.bit_count() for m in range(64) if m.bit_count()>=2)==243

def test_critical_values():
    assert abs(audit.Z_243-3.71183597358509)<1e-10
    assert abs(audit.Z_57-3.327212992073812)<1e-10
    assert 3.9<audit.Z_729<4.1

def test_bounds_agree_with_archived_p2_pair():
    for lib in ('L2','L3'):
        vectors,metadata,pair=audit.load(REPO,lib)
        historic=json.loads((REPO/f'p2/analysis/{lib}/results.json').read_text())
        diffs=[audit.paired_newcombe(vectors[pair],vectors[r],audit.Z_BASE)
               for r in (0,2,16)]
        assert abs(min(x[0] for x in diffs)-historic['pair']['excess_point'])<1e-10
        assert abs(min(x[1] for x in diffs)-historic['pair']['excess_lower'])<2e-5

def test_reconciled_counts():
    j=json.loads((ROOT/'results/inference_summary.json').read_text())['libraries']
    assert [(j[s]['counts']['family243']['POSITIVE']) for s in ('P1','L2','L3')]==[10,16,14]
    assert [(j[s]['counts']['strong_margin243']['POSITIVE']) for s in ('P1','L2','L3')]==[8,16,9]
    assert [(j[s]['counts']['hoeffding_243']['POSITIVE']) for s in ('P1','L2','L3')]==[5,8,8]
    assert [(j[s]['counts']['family729']['POSITIVE']) for s in ('P1','L2','L3')]==[9,16,14]
    assert all(all(j[s]['planted_pair_labels'][k]=='POSITIVE' for k in ('original','family57','family243','strong_margin243','hoeffding_243','family729')) for s in ('P1','L2','L3'))
    assert j['P1']['planted_pair_labels']['hoeffding_strong_margin']=='INCONCLUSIVE'
    assert j['L3']['planted_pair_labels']['hoeffding_strong_margin']=='INCONCLUSIVE'

def test_reconciled_covers_every_contrast():
    with open(ROOT/'results/contrasts_243.csv') as f:rows=list(csv.DictReader(f))
    assert len(rows)==3*243
    assert all(float(r['lower_243'])<=float(r['difference'])<=float(r['upper_243']) for r in rows)

def test_reconciled_matches_run_postaudit_labels():
    before={ (r['library'],r['mask']):r for r in csv.DictReader(open(ROOT/'results/decisions_corrected.csv'))}
    after={ (r['library'],r['mask']):r for r in csv.DictReader(open(ROOT/'results/decisions_reconciled.csv'))}
    assert len(before)==len(after)==192
    for k,r in before.items():
        assert after[k]['original']==r['original']
        assert after[k]['family243']==r['simultaneous_243']
        assert after[k]['strong_margin243']==r['strong_margin_243']
