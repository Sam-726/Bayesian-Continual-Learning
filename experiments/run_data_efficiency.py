import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
fractions=[1.0,0.5,0.25,0.10,0.05]
for f in fractions:
    print(f"\n=== DATA FRACTION {f} ===")
    subprocess.run([sys.executable,str(ROOT/"experiments"/"run_all.py"),"--data-fraction",str(f),"--seeds","1","2","3"],check=True)
