from pathlib import Path
from cluster_sensitivity import load, evaluate
from gate_forensics import analyze

ROOT=Path(__file__).resolve().parents[1]

def test_bank_alignment_and_baseline_excess():
    for lib,expected in [('P1',.53),('L2',.65),('L3',.48)]:
        bank,vec,pair=load(ROOT,lib)
        assert len(bank)==100
        assert abs((sum(vec[pair])-sum(vec[0]))/100-expected)<1e-8

def test_topic_bootstrap_reproducible_and_positive():
    bank,vec,pair=load(ROOT,'L3')
    a=evaluate(bank,vec,pair,'topic',250,77)
    b=evaluate(bank,vec,pair,'topic',250,77)
    assert a==b
    assert len(a)==3 and min(r['percentile_lower_95'] for r in a)>.15

def test_gate_infeasibility_and_frozen_fallback():
    data=analyze(ROOT)
    assert {lib:data[lib]['evaluated_attempt'] for lib in data}=={'L2':1,'L3':1}
    assert all(not r['gate_effect_feasible'] for v in data.values() for r in v['all_attempts'])
    assert not data['L3']['frozen_hypotheses']['primary_H1_and_H2']
