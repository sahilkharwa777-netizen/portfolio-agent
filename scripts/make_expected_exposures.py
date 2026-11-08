"""Generates expected_exposures.json - the INDEPENDENT golden answers for the exposure tool.

It deliberately uses only pandas on the raw CSVs (no SQLite, no tools.py), so the
evaluator and the tests compare the tool against a separately-derived result rather
than against itself. Run:  python scripts/make_expected_exposures.py
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = lambda n: pd.read_csv(os.path.join(ROOT, "data", f"{n}.csv"))

portfolios, holdings, securities, sectors = d("portfolios"), d("holdings"), d("securities"), d("sectors")
stocks = (holdings.merge(securities, on="security_id")
                  .query("asset_type == 'Stock'")
                  .merge(sectors, on="sector_id"))

golden = {}
for _, p in portfolios.iterrows():
    sub = stocks[stocks.portfolio_id == p.portfolio_id]
    total = float(sub.current_weight.sum())
    by_sector = sub.groupby("sector_name").current_weight.sum()
    golden[p.portfolio_name] = {
        "equity_weight_pct": round(total * 100, 2),
        "exposure_pct": {s: round(float(w) / total * 100, 2) for s, w in by_sector.items()} if total else {},
        "portfolio_weight_pct": {s: round(float(w) * 100, 2) for s, w in by_sector.items()},
    }

out = os.path.join(ROOT, "expected_exposures.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(golden, f, indent=2, sort_keys=True)
print(f"wrote {out} ({len(golden)} portfolios)")
