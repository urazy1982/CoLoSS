from pathlib import Path
import csv,json,sys,argparse
ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
root=a.repo.resolve();out=a.out.resolve();sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_postaudit import get_data, decide,Z95,Z57,Z243
sens=list(csv.DictReader((out/'utility_sensitivity.csv').open()))
report={};rows=[]
for lib in ('P1','L2','L3'):
    vec,meta=get_data(root,lib)
    orig={m:dict(x) for m,x in meta.items()};alt={m:dict(x) for m,x in meta.items()}
    for r in sens:
        if r['library']!=lib:continue
        m=int(r['mask']);n=int(r['validity_structural_alt'])
        val=n/60
        # Cannot auto-clear collapsed if emptiness is high. Preserve original collapse, and downgrade only.
        old=meta[m]['functionality'];nf='Collapsed' if val<0.25 or old=='Collapsed' else 'Degraded' if val<0.75 else 'Functional'
        alt[m]['functionality']=nf
        rows.append({'library':lib,'mask':m,'original_functionality':old,'sensitivity_functionality':nf,'validity_alt':n})
    a=decide(vec,orig,Z243);b=decide(vec,alt,Z243)
    changes=[{'mask':m,'old':a[m],'new':b[m]} for m in a if a[m]!=b[m]]
    fchanges=[x for x in rows if x['library']==lib and x['original_functionality']!=x['sensitivity_functionality']]
    report[lib]={'security_changes_under_structural_sensitivity':changes,'functionality_changes':fchanges,'planted_pair_old':a[48 if lib=='P1' else 18],'planted_pair_new':b[48 if lib=='P1' else 18]}
(out/'label_impact.json').write_text(json.dumps({'caveat':'Post-hoc stricter structural sensitivity, not the frozen scorer. Changes to functionality are conservative downgrades; no change to boundary output evidence.', 'libraries':report},indent=2)+'\n')
print(json.dumps(report,indent=2))
