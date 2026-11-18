"""
Paired model comparisons and robustness checks for the 5-year (and other) irrigation-intensity forecasts.
Usage (Colab, in the project folder):
    !python robustness_checks.py --panel study_outputs/panel_with_covariates.csv
Needs irrigation_pipeline.py and study_extensions.py in the same folder.
"""
import argparse, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import irrigation_pipeline as ip, study_extensions as se

warnings.filterwarnings("ignore")
K, SEED = ["state_code", "district_code"], 42


def run(d, h=5, target="irrigation_intensity"):
    d = d.sort_values(K + ["year_start"]).copy()
    g = d.groupby(K)
    cov = [c for c in d.columns if c.startswith(("rain_", "gw_", "monsoon_deficit")) and c != "rain_state_filled"]
    d["slope8"] = se._own_slope(d, target)
    d["y"] = g[target].shift(-h)
    base = [c for c in ip._feature_cols(d)
            if not c.startswith(("y_", "slope8", "future_", "rain_", "gw_", "monsoon_deficit")) and c != "y"]
    sub = d.dropna(subset=["y", target])
    ty = sub.year_start + h
    tr, te = sub[ty <= 2017], sub[ty >= 2021]
    chg = (tr["y"] - tr[target]).groupby(tr.state_code).mean()
    P = {"persistence": te[target].values,
         "state_mean_change": (te[target] + te.state_code.map(chg).fillna(0)).values}
    for name, cols in [("gbm_base", base + ["slope8"]), ("gbm_plus_cov", base + ["slope8"] + cov)]:
        cols = [c for c in dict.fromkeys(cols) if tr[c].nunique(dropna=True) >= 2]   # drop empty/constant columns
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, l2_regularization=1.0,
                                          early_stopping=True, random_state=SEED)
        m.fit(tr[cols], tr["y"] - tr[target])
        P[name] = m.predict(te[cols]) + te[target].values
    return te, P


def paired(te, P, a, b, n=1000):
    """skill of model a relative to model b (1 - MAE_a/MAE_b) with a district-level bootstrap interval."""
    y = te["y"].values
    dist = (te.state_code.astype(str) + "_" + te.district_code.astype(str)).values
    t = pd.DataFrame({"g": dist, "a": np.abs(y - P[a]), "b": np.abs(y - P[b])}).groupby("g").sum()
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(t))
    v = [1 - t.a.values[s].sum() / t.b.values[s].sum() for s in (rng.choice(idx, len(idx)) for _ in range(n))]
    return 1 - t.a.sum() / t.b.sum(), np.percentile(v, [2.5, 97.5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="study_outputs/panel_with_covariates.csv")
    a = ap.parse_args()
    full = pd.read_csv(a.panel)
    rows = []
    print("== Full sample, h=5, paired comparisons")
    te, P = run(full, 5)
    for x, y in [("gbm_plus_cov", "state_mean_change"), ("gbm_plus_cov", "gbm_base"),
                 ("gbm_plus_cov", "persistence"), ("state_mean_change", "persistence")]:
        pt, ci = paired(te, P, x, y)
        print(f"  {x:18s} vs {y:18s} {pt:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        rows.append(dict(sample="full", horizon=5, model=x, vs=y, skill=pt, lo=ci[0], hi=ci[1]))
    if "rain_state_filled" in full:
        sub = full[full.rain_state_filled == 0]
        print(f"\n== Excluding state-filled rainfall rows ({len(sub)} of {len(full)} retained)")
        for h in (1, 3, 5):
            te, P = run(sub, h)
            for x in ["state_mean_change", "gbm_base", "gbm_plus_cov"]:
                pt, ci = paired(te, P, x, "persistence", n=500)
                print(f"  h={h} {x:18s} vs persistence {pt:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
                rows.append(dict(sample="no_state_fill", horizon=h, model=x, vs="persistence", skill=pt, lo=ci[0], hi=ci[1]))
    pd.DataFrame(rows).to_csv("study_outputs/robustness_results.csv", index=False)
    print("\nSaved study_outputs/robustness_results.csv")
