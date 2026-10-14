"""One canonical artifact feeds paper, notebook and static dashboard exports."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def export(home):
    home=Path(home)
    records=[json.loads(p.read_text()) for p in (home/"artifacts").glob("*/result.json")]
    out=home/"data/processed"
    out.mkdir(parents=True,exist_ok=True)
    figdir=home/"paper/figures"
    figdir.mkdir(parents=True,exist_ok=True)
    rows=[{"id":r["id"],"source":r["dataset"]["source"],"development_sample":r["dataset"]["development_sample"],"demonstration":r["dataset"]["demonstration"],"split_hash":r["split_hash"],"method":r["configuration"]["method"],"seed":r["configuration"]["seed"],**r["metrics"]} for r in records]
    df=pd.DataFrame(rows)
    df.drop(columns=["confusion_matrix"],errors="ignore").to_csv(out/"comparison.csv",index=False)
    (out/"dashboard.json").write_text(json.dumps(records,indent=2))
    # Choose most recent controlled IEEE-CIS group; never mix dataset/split variants.
    real=[r for r in records if not r["dataset"]["demonstration"] and r["configuration"]["method"] in ["baseline","smote","cost_sensitive"] and r["configuration"].get("run_role","primary")=="primary"]
    chosen=[]
    if real:
        latest=max(real,key=lambda r:(home/"artifacts"/r["id"]/"result.json").stat().st_mtime)
        chosen=[r for r in real if r["split_hash"]==latest["split_hash"] and r["dataset"]["fingerprint"]==latest["dataset"]["fingerprint"]]
    (home/"paper/selected_runs.json").write_text(json.dumps([r["id"] for r in chosen]))
    table=[r"\begin{tabular}{lrrrr}",r"\hline Method & AP & Precision & Recall & F1 \\",r"\hline"]
    for method in ["baseline","smote","cost_sensitive"]:
        subset=[r for r in chosen if r["configuration"]["method"]==method]
        if subset:
            vals=np.array([[r["metrics"][k] for k in ["average_precision","precision","recall","f1"]] for r in subset])
            mean,std=vals.mean(0),vals.std(0,ddof=1) if len(vals)>1 else np.zeros(4)
            label={"baseline":"Baseline ANN","smote":"SMOTENC + ANN","cost_sensitive":"Weighted ANN"}[method]
            formatted=[f"{m:.3f}"+(f" $\\pm$ {s:.3f}" if len(vals)>1 else "") for m,s in zip(mean,std)]
            table.append(label+" & "+" & ".join(formatted)+r" \\")
    table.extend([r"\hline",r"\end{tabular}"])
    (home/"paper/results_table.tex").write_text("\n".join(table))
    caption="IEEE-CIS results unavailable; results section incomplete."
    if chosen:
        first=chosen[0]
        caption=f"IEEE-CIS {'development sample' if first['dataset']['development_sample'] else 'full labeled data'}, {sum(first['partition_counts'].values()):,} observations; {first['partition_counts']['test']:,} fixed test observations. Separate initialization seeds share one split."
    (home/"paper/results_status.tex").write_text(caption)
    if chosen:
        fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout="constrained")
        for method in ["baseline","smote","cost_sensitive"]:
            subset=[r for r in chosen if r["configuration"]["method"]==method]
            if not subset:
                continue
            r=sorted(subset,key=lambda r:r["configuration"]["seed"])[0]
            for ax,key in zip(axes,["roc","pr"]):
                points=r["curves"][key]
                ax.plot([p["x"] for p in points],[p["y"] for p in points],label=method.replace("_"," "))
        axes[0].plot([0,1],[0,1],"--",color="gray",linewidth=.8)
        axes[0].set(xlabel="False-positive rate",ylabel="True-positive rate",title="ROC (first recorded seed)")
        axes[1].axhline(chosen[0]["metrics"]["prevalence"],ls="--",color="gray",lw=.8)
        axes[1].set(xlabel="Recall",ylabel="Precision",title="Precision-recall")
        axes[1].legend(fontsize=7)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.savefig(figdir/"comparison.pdf")
        fig.savefig(figdir/"comparison.png",dpi=220)
        plt.close(fig)
    print(f"Exported {len(records)} saved runs; selected {len(chosen)} controlled IEEE-CIS runs")
    return chosen

if __name__=="__main__":
    export(Path(__file__).resolve().parents[1])
