import csv
import os
from pathlib import Path

results_dir = Path(__file__).parent / "results"

rows = []
for path in sorted(results_dir.glob("*.csv")):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rmse_values = [float(row["RMSE"]) for row in reader]
    avg_rmse = sum(rmse_values) / len(rmse_values)
    rows.append((path.name, avg_rmse, len(rmse_values)))

rows.sort(key=lambda x: x[1])

print(f"{'Rank':<6} {'Avg RMSE':<12} {'Windows':<10} {'File'}")
print("-" * 70)
for rank, (name, avg_rmse, n_windows) in enumerate(rows, start=1):
    print(f"{rank:<6} {avg_rmse:<12.4f} {n_windows:<10} {name}")
