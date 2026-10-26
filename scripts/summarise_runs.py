import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
rows = []
for p in sorted(root.glob("*/summary.json")):
    s = json.loads(p.read_text())
    rows.append((p.parent.name, s["n"], s["success"], s["LER"], s["mean_env_steps"], s["TF"]))
print("| run | n | success | LER | env steps | TF |")
print("|---|---|---|---|---|---|")
for name, n, su, le, st, tf in rows:
    print(f"| {name} | {n} | {su:.2f} | {le:.2f} | {st:.1f} | {tf:.2f} |")
