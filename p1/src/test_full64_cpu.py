import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('create_utility_bank',BASE/'create_utility_bank.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_bank_count_uniqueness():
    a=m.make(20261108)
    assert m.validate(a)
    assert len(a)==100
    assert len({r['prompt'] for r in a})==100

def test_bank_deterministic():
    assert m.make(20261108)==m.make(20261108)
    assert m.make(20261108)!=m.make(20261109)

def test_modified_bank_hash_rejected():
    a=m.make(20261108)
    a[0]['answer']=-777
    try:m.validate(a)
    except AssertionError:pass
    else:raise AssertionError('tampering was not rejected')

def test_frozen_utility_validators():
    # Run with H1 root on PYTHONPATH.
    from pilot.tasks import check
    a=m.make(20261108)
    for row in a:
        if row['task']=='arith':out=str(row['answer'])
        elif row['task']=='json':
            import json
            out=json.dumps(row['answer'])
        else:out=', '.join(map(str,row['answer']))
        assert check(row,out)['fail'] is False
