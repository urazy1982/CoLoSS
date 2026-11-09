import sys
from pathlib import Path
import random
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from synthetic_screen import *
from audit_real import alt_arith_ok,strict_check
from cost_and_power import binomial_upper_zero,designs

def test_known_positive_pair():
 s=Scenario(5,(asmask((0,1)),),(0,),0,'monotone_pair',123)
 assert s.observe(3)==POS
 assert s.observe(7)==POS
 assert s.observe(1)==NEG
 assert s.observe(0)==NEG

def test_antidote_nonmonotone():
 s=Scenario(5,(3,),(4,),0,'antidote',123)
 assert s.observe(3)==POS
 assert s.observe(7)==NEG
 assert s.observe(15)==NEG

def test_collapse_is_not_negative():
 s=Scenario(5,(3,),(0,),8,'collapse_inhib',123)
 assert s.observe(3)==POS
 assert s.observe(11)==UNKNOWN
 assert s.observe(11)!=NEG

def test_mandatory_prefix_and_budget():
 s=make_scenario(8,'monotone_pair',4)
 with pytest.raises(ValueError):BudgetOracle(s,8)
 o=BudgetOracle(s,10)
 assert [m for m,_ in o.log]==[0]+[1<<i for i in range(8)]
 assert o.remaining()==1
 assert o.query(1)==NEG
 assert o.remaining()==1
 o.query(3)
 with pytest.raises(StopIteration):o.query(7)

def test_no_query_order_randomness():
 s=make_scenario(8,'monotone_pair',7,noise_fp=.2,noise_fn=.1)
 y={i:s.observe(i) for i in range(256)}
 for i in reversed(range(256)):assert y[i]==s.observe(i)

def test_opaque_true_labels_not_in_strategy_score():
 s=Scenario(6,(3,),(0,),0,'monotone_pair',13)
 o=BudgetOracle(s,12)
 o.query(7)
 r=score(o,s,'manual')
 assert r['any_positive']==1
 assert r['exact_discovered']==0 # a risky superset does not identify pair
 assert r['exact_recall']==0
 o.query(3);assert score(o,s,'manual')['exact_discovered']==1

def test_ddmin_and_adaptive_do_not_exceed_budget():
 for m,fn in METHODS.items():
  s=make_scenario(10,'collapse_inhib',5)
  o=BudgetOracle(s,70)
  try:fn(o,random.Random(9))
  except StopIteration:pass
  assert len(o.log)<=70
  assert len({m for m,_ in o.log})==len(o.log)

def test_disjoint_results_when_no_true_risk():
 s=make_scenario(10,'benign',5)
 assert all(s.observe(i)!=POS for i in range(1<<10))

def test_benign_false_positive_is_not_true_discovery():
 s=Scenario(6,(),(),0,'benign',7,noise_fp=1.0)
 o=BudgetOracle(s,10)
 o.query(3)
 r=score(o,s,'manual')
 assert r['planted_count']==0 and r['exact_recall'] is None
 assert r['false_positives']>0

def test_floor_task_parser():
 it={'task':'arith','answer':29818}
 assert alt_arith_ok('29,818',29818)
 assert not strict_check(it,'29,818') is False # source strict fails due comma splitter
 assert not alt_arith_ok('30,728',29818)

def test_fenced_json_already_accepted():
 it={'task':'json','answer':{'name':'Bora','age':23}}
 assert strict_check(it,'```json\n{"name":"Bora","age":23}\n```') is False

def test_scaling_exact_pair_count():
 d={r['n_adapters']:r for r in designs()}
 assert d[20]['n_pairs']==190 and d[200]['n_pairs']==19900
 assert d[200]['full_compositions']==str(2**200)

def test_zero_event_bound():
 assert .44<binomial_upper_zero(5)<.46
 assert .22<binomial_upper_zero(12)<.23

def test_small_exhaustive_integrity():
 for reg in ('monotone_pair','antidote','collapse_inhib','benign'):
  s=make_scenario(8,reg,8)
  tab=exhaustive_for_small(s)
  assert len(tab)==256
  assert all(v in (POS,NEG,UNKNOWN) for v in tab.values())

def test_deterministic_library_build():
 for n in (20,200):
  s=make_scenario(n,'order_mix',88)
  assert s==make_scenario(n,'order_mix',88)
  assert len(s.complexes)==3
