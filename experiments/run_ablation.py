# Lightweight beta ablation. Results are written to separate folders.
import subprocess, sys, yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=yaml.safe_load(open(ROOT/"configs/default.yaml"))
for beta in [0.0001,0.001,0.01,0.1]:
    cfg=ROOT/f"configs/beta_{beta}.yaml"
    c=dict(base); c["kl_beta"]=beta; c["results_root"]=str(ROOT/f"results/beta_{beta}")
    with open(cfg,"w") as f: yaml.safe_dump(c,f)
    subprocess.run([sys.executable,str(ROOT/"experiments"/"run_all.py"),"--config",str(cfg),"--seeds","1","2","3"],check=True)
