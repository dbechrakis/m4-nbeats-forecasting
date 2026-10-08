"""Check committed results for internal consistency without retraining."""

import ast
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for path in ROOT.rglob("*.py"):
    if not any(part in path.parts for part in (".git", ".venv")):
        ast.parse(path.read_text())

for directory in sorted((ROOT / "outputs").glob("*/")):
    metrics_path = directory / "metrics.csv"
    if not metrics_path.exists():
        continue
    with metrics_path.open() as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    naive2 = rows["Naive2"]
    assert math.isclose(float(naive2["owa"]), 1.0, abs_tol=1e-4), directory
    for row in rows.values():
        expected = 0.5 * (float(row["smape"]) / float(naive2["smape"]) + float(row["mase"]) / float(naive2["mase"]))
        assert math.isclose(float(row["owa"]), expected, abs_tol=2e-4), (directory, row["method"])
    summary = json.loads((directory / "summary.json").read_text())
    members = sorted((directory / "members").glob("*.json"))
    assert summary["ensemble_members"] == len(members) == len(summary["member_validation_smape"]), directory
    for member in members:
        record = json.loads(member.read_text())
        steps = record["curve"][-1]["step"]
        assert 0 < record["best_step"] <= steps
        assert math.isclose(min(c["validation_smape"] for c in record["curve"]), record["validation_smape"], rel_tol=1e-6)
    assert math.isclose(summary["owa_ensemble"], float(rows["N-BEATS ensemble (median)"]["owa"]), abs_tol=1e-4)

print("Committed M4 results are internally consistent.")
