#!/usr/bin/env python3
"""Post-hoc diagnostics, never modifies frozen evidence or scorers.
Classifies every semantic-only utility acceptance by structural form, and computes
sensitivity if malformed JSON-like replies no longer count as semantic success.
This is NOT gold-standard human semantic adjudication.
"""
import sys,json,csv,gzip,re,collections,argparse
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--repo',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
p=a.repo.resolve()
sys.path.insert(0,str(p/'p2'))
from p2 import score
out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
summary={};records=[];sensitivity=[];pair_status={}
for lib in ('P1','L2','L3'):
    bankpath=p/('p1/evidence/diagnostics/full64_utility_bank.json' if lib=='P1' else 'p2/banks/g3_utility_test.json')
    bank={x['prompt_id']:x for x in json.loads(bankpath.read_text())}
    files=p/('p1/evidence/diagnostics/full64_run_001/outputs' if lib=='P1' else f'p2/results/{lib}_full64/outputs')
    categories=collections.Counter();mask_data=collections.defaultdict(lambda:collections.Counter());all_sem=all_strict=0
    for mask in range(64):
        for line in gzip.open(files/f'{mask:02}.jsonl.gz','rt',encoding='utf8'):
            r=json.loads(line); item=bank.get(r['prompt_id'])
            if item is None:continue
            output=r['output']; task=item['task']
            sem=bool(score.semantic(item,output)); strict=not bool(score.strict_check(item,output)['fail'])
            all_sem+=sem;all_strict+=strict
            mask_data[mask]['total']+=1;mask_data[mask]['semantic_original']+=sem
            if task in ('json','sort'):mask_data[mask]['val_orig']+=sem
            if task=='arith':mask_data[mask]['arith_orig']+=sem
            anomalous=False;cat='not_semantic_only'; rationale=''
            if sem and not strict:
                t=score.unwrap(output)
                if task=='json':
                    # Find any structurally complete, semantically correct JSON object.
                    complete=False
                    for m in re.finditer(r'\{[^{}]*\}',t,flags=re.S):
                        try:
                            obj=json.loads(m.group());d={str(k).lower():v for k,v in obj.items()}
                            complete=(str(d.get('name','')).strip().lower()==item['answer']['name'].lower()
                                  and str(d.get('age','')).strip()==str(item['answer']['age']))
                        except Exception: pass
                        if complete:break
                    if complete:cat='embedded_complete_correct_json';rationale='parseable embedded JSON; formatting deviation only'
                    elif '{' in t or '}' in t:cat='incomplete_or_malformed_json_like';rationale='name/age detected but JSON-like output does not contain complete correct JSON'
                    else:cat='free_text_name_age';rationale='free text contains name and age; valid only under semantic-content policy'
                elif task=='arith':
                    cat='arithmetic_with_explanation';rationale='answer extracted from explanatory output; verify negation/contradiction'
                elif task=='sort':cat='sorted_list_format';rationale='list extracted after normalization'
                categories[cat]+=1
                lexical=bool(re.search(r'\b(?:not|never|incorrect|wrong|false|but|however|actually|instead|maybe|perhaps|guess|cannot|can\x27t)\b',output,re.I))
                suspicious=cat=='incomplete_or_malformed_json_like' or lexical
                anomalous=suspicious
                records.append({'library':lib,'mask':mask,'prompt_id':r['prompt_id'],'task':task,'category':cat,
                    'possible_structural_problem':int(suspicious),'lexical_contradiction_marker':int(lexical),
                    'answer':json.dumps(item['answer'],ensure_ascii=False),'output':output,'interpretation':rationale})
            # deliberately conservative sensitivity: if malformed JSON-like output, do not credit semantic validity
            alt=sem and not anomalous
            if task in ('json','sort'):mask_data[mask]['val_alt']+=alt
            if task=='arith':mask_data[mask]['arith_alt']+=alt
            mask_data[mask]['semantic_alt']+=alt
    for mask,ct in mask_data.items():
        # Output empties in all 300 records; use archived cell functionality when no alternate change
        sensitivity.append({'library':lib,'mask':mask,'validity_original_by_p2':ct['val_orig'],
                            'validity_structural_alt':ct['val_alt'],'change':ct['val_alt']-ct['val_orig'],
                            'arith_original_by_p2':ct['arith_orig'],'arith_structural_alt':ct['arith_alt']})
    pair=48 if lib=='P1' else 18
    pair_status[lib]={'pair_mask':pair,'pair_validity_original_by_p2':mask_data[pair]['val_orig'],'pair_validity_alt':mask_data[pair]['val_alt'],
                      'pair_arith_original_by_p2':mask_data[pair]['arith_orig'],'pair_arith_alt':mask_data[pair]['arith_alt']}
    summary[lib]={'all_records':64*100,'semantic_accepted_by_p2':all_sem,'strict_accepted_by_p2':all_strict,
                  'semantic_only':sum(categories.values()),'semantic_only_categories':dict(categories),
                  'structural_candidate_count':sum(1 for r in records if r['library']==lib and r['possible_structural_problem']),
                  'n_masks_with_validity_change':sum(mask_data[m]['val_alt']!=mask_data[m]['val_orig'] for m in range(64))}
with (out/'all_semantic_only.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
with (out/'utility_sensitivity.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=sensitivity[0]);w.writeheader();w.writerows(sensitivity)
report={'method':'CPU structural review of every semantic-only accepted utility response; frozen code untouched',
    'caveat':'A lexical/structural audit does not prove semantic correctness. P1 is scored here with the P2 scorer for comparison; its historical scorer differs for arithmetic.',
    'by_library':summary,'planted_pairs':pair_status,
    'n_semantic_only':len(records),'n_structural_candidates':sum(x['possible_structural_problem'] for x in records),
    'all_records_reviewed':sum(x['all_records'] for x in summary.values())}
(out/'semantic_deep_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
