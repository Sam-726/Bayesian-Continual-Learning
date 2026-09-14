from __future__ import annotations
import sys, os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import get_split_mnist, make_loader
from src.models import MLP, BayesianMLP
from src.train import set_seed, device_from_config, train_deterministic, train_bayesian, evaluate_deterministic, evaluate_bayesian
from src.metrics import forgetting, ece

def subset_fraction(dataset, fraction, seed):
    if fraction >= 1.0: return dataset
    g = np.random.default_rng(seed)
    n = max(1, int(len(dataset)*fraction))
    idx = g.choice(len(dataset), n, replace=False).tolist()
    return torch.utils.data.Subset(dataset, idx)

def run(method, cfg, tasks, seed):
    device=device_from_config(cfg["device"])
    hidden1,hidden2=cfg["hidden1"],cfg["hidden2"]
    model = MLP(hidden1,hidden2) if method=="baseline" else BayesianMLP(hidden1,hidden2)
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    matrix=[]
    task_times=[]
    confidence_final=[]; correct_final=[]; entropy_final=[]
    for t,task in enumerate(tasks):
        trainset=subset_fraction(task.train,cfg["data_fraction"],seed+t)
        train_loader=make_loader(trainset,cfg["batch_size"],True,cfg["num_workers"])
        start=__import__("time").time()
        if method=="baseline":
            train_deterministic(model,train_loader,optimizer,device,cfg["epochs"])
        else:
            train_bayesian(model,train_loader,optimizer,device,cfg["epochs"],cfg["kl_beta"])
            model.update_priors()
        task_times.append(__import__("time").time()-start)
        row=[]
        for j in range(t+1):
            test_loader=make_loader(tasks[j].test,cfg["batch_size"],False,cfg["num_workers"])
            if method=="baseline":
                acc,conf,corr=evaluate_deterministic(model,test_loader,device)
                ent=np.zeros_like(conf)
            else:
                acc,conf,corr,ent=evaluate_bayesian(model,test_loader,device,cfg["posterior_samples"])
            row.append(acc)
            if t==len(tasks)-1:
                confidence_final.extend(conf); correct_final.extend(corr); entropy_final.extend(ent)
        row += [None]*(len(tasks)-len(row))
        matrix.append(row)
    return {"method":method,"seed":seed,"matrix":matrix,
            "forgetting":forgetting(matrix),"task_times":task_times,
            "ece":ece(confidence_final,correct_final),
            "final_entropy_mean":float(np.mean(entropy_final)) if entropy_final else 0.0}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default=str(ROOT/"configs/default.yaml"))
    ap.add_argument("--seeds",nargs="+",type=int,default=[1,2,3])
    ap.add_argument("--data-fraction",type=float,default=None)
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config))
    if args.data_fraction is not None: cfg["data_fraction"]=args.data_fraction
    set_seed(cfg["seed"])
    tasks=get_split_mnist(cfg["data_root"],cfg["num_workers"])
    all_results=[]
    for seed in args.seeds:
        set_seed(seed)
        for method in ("baseline","bayesian"):
            print(f"Running {method}, seed={seed}, fraction={cfg['data_fraction']}")
            all_results.append(run(method,cfg,tasks,seed))
    out=Path(cfg["results_root"]); (out/"tables").mkdir(parents=True,exist_ok=True); (out/"figures").mkdir(parents=True,exist_ok=True)
    with open(out/"tables"/"raw_results.json","w") as f: json.dump(all_results,f,indent=2)
    rows=[]
    for r in all_results:
        final=[x for x in r["matrix"][-1] if x is not None]
        rows.append({"method":r["method"],"seed":r["seed"],"final_avg_accuracy":np.mean(final),"forgetting":r["forgetting"],"ece":r["ece"],"final_entropy_mean":r["final_entropy_mean"]})
    df=pd.DataFrame(rows); df.to_csv(out/"tables"/"summary.csv",index=False)
    print("\nSummary by method:")
    print(df.groupby("method")[["final_avg_accuracy","forgetting","ece"]].agg(["mean","std"]))
    summary=df.groupby("method").agg({"final_avg_accuracy":["mean","std"],"forgetting":["mean","std"]})
    methods=list(summary.index)
    plt.figure(figsize=(7,4))
    vals=[summary.loc[m,("final_avg_accuracy","mean")] for m in methods]
    errs=[summary.loc[m,("final_avg_accuracy","std")] for m in methods]
    plt.bar(methods,vals,yerr=errs,capsize=4)
    plt.ylabel("Final average accuracy")
    plt.title("Bayesian vs sequential fine-tuning")
    plt.tight_layout(); plt.savefig(out/"figures"/"final_accuracy.png",dpi=180); plt.close()
    plt.figure(figsize=(7,4))
    vals=[summary.loc[m,("forgetting","mean")] for m in methods]
    errs=[summary.loc[m,("forgetting","std")] for m in methods]
    plt.bar(methods,vals,yerr=errs,capsize=4)
    plt.ylabel("Average forgetting")
    plt.title("Catastrophic forgetting")
    plt.tight_layout(); plt.savefig(out/"figures"/"forgetting.png",dpi=180); plt.close()

if __name__=="__main__":
    main()
