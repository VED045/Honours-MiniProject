import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.export_results import export
home=Path(__file__).resolve().parents[1]
chosen=export(home)
paper=home/'paper'
names={'baseline':'Baseline ANN','smote':'SMOTENC + ANN','cost_sensitive':'Weighted ANN'}
if not chosen:
    for name in ['setup_table','counts_table','results_narrative','development_narrative']:
        (paper/f'{name}.tex').write_text('Results-dependent content incomplete. No compatible IEEE-CIS artifacts available.')
    raise SystemExit(0)
c=chosen[0]['configuration']
setup=[r'\begin{tabular}{ll}',r'\toprule Parameter & Value \\',r'\midrule']
for name,val in [('Hidden layers',str(c['hidden_layers'])),('Activation / dropout',f"{c['activation']} / {c['dropout']}"),('Batch normalization',str(c['batch_norm'])),('Optimizer / learning rate',f"{c['optimizer']} / {c['learning_rate']}"),('Epoch limit / patience',f"{c['epochs']} / {c['patience']}"),('Batch size',str(c['batch_size'])),('L1 / weight decay',f"{c['l1']} / {c['l2']}"),('Class weights (weighted only)',f"{c['legitimate_weight']} / {c['fraud_weight']}"),('Oversampling ratio / neighbors',f"{c['sampling_ratio']} / {c['neighbors']}"),('Split seed',str(c['split_seed'])),('Numerical processing',f"{c['numerical_imputation']} / {c['scaling']}"),('Original feature subset',str(len(c['features']))+' predictors')]:
    setup.append(name+' & '+val+r' \\')
setup.extend([r'\bottomrule',r'\end{tabular}'])
(paper/'setup_table.tex').write_text('\n'.join(setup))
table=[r'\begin{tabular}{lrrrrr}',r'\toprule Method & $\tau$ & FP & FN & Train (s) & Infer (s) \\',r'\midrule']
paragraphs=[]
for method in names:
    subset=[r for r in chosen if r['configuration']['method']==method]
    if not subset:continue
    first=sorted(subset,key=lambda r:r['configuration']['seed'])[0]
    m=first['metrics']
    table.append(f"{names[method]} & {m['threshold']:.2f} & {m['fp']} & {m['fn']} & {m['training_seconds']:.1f} & {m['inference_seconds']:.3f}"+r' \\')
    mean=np.mean([r['metrics']['average_precision'] for r in subset])
    paragraphs.append(f"{names[method]} records {'mean ' if len(subset)>1 else ''}AP {mean:.4f}. For the first seed, precision is {m['precision']:.4f}, recall is {m['recall']:.4f}, and F1 is {m['f1']:.4f} at validation-selected cutoff {m['threshold']:.2f}. It flags {m['flagged']:,} of {m['transactions']:,} test transactions, including {m['fp']:,} false positives, while missing {m['fn']:,} frauds.")
    ci=first['bootstrap_interval'].get('average_precision')
    if ci:paragraphs.append(f"The IID test AP interval is [{ci['low']:.4f}, {ci['high']:.4f}] from {ci['replicates']} valid bootstrap replicates; temporal dependence is not modeled.")
table.extend([r'\bottomrule',r'\end{tabular}'])
(paper/'counts_table.tex').write_text('\n'.join(table))
first=chosen[0]
ranked=sorted([(method,np.mean([r['metrics']['average_precision'] for r in chosen if r['configuration']['method']==method])) for method in names if any(r['configuration']['method']==method for r in chosen)],key=lambda x:x[1],reverse=True)
paragraphs.insert(0,f"The test fraud prevalence is {first['metrics']['prevalence']*100:.3f}\\%. The completed group contains {len(chosen)} runs on a common fingerprint and split hash. {names[ranked[0][0]]} has the highest recorded AP in this group; this internal descriptive result does not establish general superiority.")
paragraphs.append(f"Exact-covariate duplicate removal excluded {first['dataset']['duplicates_removed']:,} observations. The split has {first['partition_counts']['train']:,} training, {first['partition_counts']['validation']:,} validation, and {first['partition_counts']['test']:,} test observations. Figures and counts come from saved probabilities and metadata.")
(paper/'results_narrative.tex').write_text('\n\n'.join(paragraphs))
development=[json.loads(p.read_text()) for p in (home/'artifacts').glob('*/result.json')]
development=[r for r in development if r['dataset']['development_sample'] and not r['dataset']['demonstration'] and r['configuration'].get('run_role','primary')=='primary']
dev=[]
for method in names:
    subset=[r for r in development if r['configuration']['method']==method]
    if not subset:continue
    vals=np.array([r['metrics']['average_precision'] for r in subset])
    dev.append(f"{names[method]} development AP is {vals.mean():.4f}"+(f" $\\pm$ {vals.std(ddof=1):.4f}" if len(vals)>1 else '')+f" across {len(vals)} seeds.")
(paper/'development_narrative.tex').write_text(' '.join(dev) or 'No development experiments recorded.')
print('Generated quantitative paper content directly from artifacts')
