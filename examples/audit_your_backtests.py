"""Audit backtests produced anywhere else (a research platform, a parameter sweep).

Export one column of daily returns per strategy you tried -- including the ones
you discarded -- then:

    skeptic audit backtests.csv --trials 400     # 400 = all ideas tried, if not all were exported

This script writes a demo CSV: 60 noise strategies plus one with a real edge.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
T = 1250
data = {f"idea_{k:02d}": rng.standard_t(4, T) * 0.004 for k in range(60)}
data["real_edge"] = rng.standard_t(4, T) * 0.004 + 0.0004
df = pd.DataFrame(data, index=pd.bdate_range("2021-01-01", periods=T, name="date"))
df.to_csv("backtests.csv")
print("wrote backtests.csv -> now run:  skeptic audit backtests.csv")
