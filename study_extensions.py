"""
Extensions to study_pipeline.py (run AFTER the base pipeline; needs irrigation_pipeline.py + study_pipeline.py)

  1. trend_analysis_robust()    trend-free pre-whitening + Benjamini-Hochberg FDR
  2. forecast_with_baselines()  persistence vs district-trend extrapolation vs state-mean-change vs
                                GBM (with and without trend features) + bootstrap CI on skill
  3. cluster_stability()        bootstrap ARI for KMeans over k, plus Gaussian mixture comparison

Usage in Colab:
    import study_extensions as se
    rob   = se.trend_analysis_robust(df, OUT)
    res   = se.forecast_with_baselines(df, OUT)
    stab  = se.cluster_stability(df, tr, OUT)
"""
import os, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
SEED = 42
KEYS = ["state_code", "district_code"]


# ------------------------------------------------------------------ 1. robust trends
def _bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (NaNs stay NaN)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def _tfpw_mk(y, x):
    """Trend-free pre-whitening Mann-Kendall (Yue et al. 2002 style). Returns tau, p."""
    m = ~np.isnan(y)
    y, x = y[m], x[m]
    if len(y) < 15:
        return np.nan, np.nan
    b = stats.theilslopes(y, x)[0]
    detr = y - b * x
    r1 = np.corrcoef(detr[:-1], detr[1:])[0, 1] if np.std(detr) > 0 else 0.0
    r1 = 0.0 if np.isnan(r1) else r1
    if abs(r1) > 2 / np.sqrt(len(y)):                      # autocorrelation significant -> pre-whiten
        resid = detr[1:] - r1 * detr[:-1]
        y_pw = resid + b * x[1:]
        return stats.kendalltau(x[1:], y_pw)
    return stats.kendalltau(x, y)


def trend_analysis_robust(df, out, indicators=("groundwater_share", "irrigation_intensity"), alpha=0.05):
    rows = []
    for (sc, dc), g in df.groupby(KEYS):
        x = g.year_start.values.astype(float)
        r = dict(state_code=sc, district_code=dc, state_name=g.state_name.iloc[0], district_name=g.district_name.iloc[0])
        for v in indicators:
            y = g[v].values.astype(float)
            m = ~np.isnan(y)
            if m.sum() < 15:
                r[f"{v}_tau"] = r[f"{v}_p_raw"] = r[f"{v}_p_tfpw"] = np.nan
                continue
            tau, p_raw = stats.kendalltau(x[m], y[m])
            _, p_pw = _tfpw_mk(y, x)
            r[f"{v}_tau"], r[f"{v}_p_raw"], r[f"{v}_p_tfpw"] = tau, p_raw, p_pw
        rows.append(r)
    t = pd.DataFrame(rows)
    summary = []
    for v in indicators:
        t[f"{v}_p_fdr"] = _bh_fdr(t[f"{v}_p_tfpw"].values)
        for tag, col in [("raw", "p_raw"), ("tfpw", "p_tfpw"), ("tfpw+fdr", "p_fdr")]:
            p = t[f"{v}_{col}"]
            sig = p < alpha
            summary.append(dict(indicator=v, method=tag,
                                rising=int((sig & (t[f"{v}_tau"] > 0)).sum()),
                                declining=int((sig & (t[f"{v}_tau"] < 0)).sum()),
                                tested=int(p.notna().sum())))
    s = pd.DataFrame(summary)
    t.to_csv(os.path.join(out, "district_trends_robust.csv"), index=False)
    s.to_csv(os.path.join(out, "trend_robustness_summary.csv"), index=False)
    print("[Robust trends]\n", s.to_string(index=False))
    return t


# ------------------------------------------------------------------ 2. forecasting with proper baselines
def _own_slope(df, col, window=8, min_obs=5):
    """OLS slope per year over the last `window` years (uses data up to and including the row's year)."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.sort_values(KEYS + ["year_start"]).groupby(KEYS):
        y, x = g[col].values, g.year_start.values.astype(float)
        sl = np.full(len(g), np.nan)
        for i in range(len(g)):
            lo = max(0, i - window + 1)
            yy, xx = y[lo:i + 1], x[lo:i + 1]
            m = ~np.isnan(yy)
            if m.sum() >= min_obs and np.ptp(xx[m]) > 0:
                sl[i] = np.polyfit(xx[m], yy[m], 1)[0]
        out.loc[g.index] = sl
    return out


def _boot_skill(y, p_model, p_base, groups, n=500, seed=SEED):
    """Cluster bootstrap (by district) of skill = 1 - MAE_model / MAE_base."""
    rng = np.random.default_rng(seed)
    ae_m, ae_b = np.abs(y - p_model), np.abs(y - p_base)
    df = pd.DataFrame({"g": groups, "m": ae_m, "b": ae_b}).groupby("g").agg(m=("m", "sum"), b=("b", "sum"), n=("m", "size"))
    idx = np.arange(len(df))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        vals.append(1 - df.m.values[s].sum() / df.b.values[s].sum())
    return np.percentile(vals, [2.5, 97.5])


def forecast_with_baselines(df, out, targets=("irrigation_intensity", "groundwater_share"), horizons=(1, 3, 5)):
    from sklearn.ensemble import HistGradientBoostingRegressor
    import irrigation_pipeline as ip

    d = df.sort_values(KEYS + ["year_start"]).copy()
    g = d.groupby(KEYS)
    for t in targets:
        d[f"slope8_{t}"] = _own_slope(d, t)
        d[f"ystate_{t}"] = d.groupby(["state_code", "year_start"])[t].transform("mean")
        for h in horizons:
            d[f"y_{t}_h{h}"] = g[t].shift(-h)
    base_cols = [c for c in ip._feature_cols(d) if not c.startswith(("y_", "slope8_", "ystate_"))]
    trend_cols = [f"slope8_{t}" for t in targets]
    rows = []
    for t in targets:
        for h in horizons:
            y = f"y_{t}_h{h}"
            sub = d.dropna(subset=[y, t])
            ty = sub.year_start + h
            train, test = sub[ty <= 2017], sub[ty >= 2021]
            # state mean change over the training window (simple, leakage-free)
            chg = (train[y] - train[t]).groupby(train.state_code).mean()
            preds = {
                "persistence": test[t].values,
                "trend_extrap(8y OLS)": (test[t] + test[f"slope8_{t}"].fillna(0) * h).clip(0, 1.2).values,
                "damped_trend(0.5)": (test[t] + 0.5 * test[f"slope8_{t}"].fillna(0) * h).clip(0, 1.2).values,
                "state_mean_change": (test[t] + test.state_code.map(chg).fillna(0)).values,
            }
            for name, cols in [("gbm_base", base_cols), ("gbm_plus_trend", base_cols + trend_cols)]:
                m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, l2_regularization=1.0,
                                                  early_stopping=True, random_state=SEED)
                m.fit(train[cols], train[y] - train[t])
                preds[name] = m.predict(test[cols]) + test[t].values
            yt = test[y].values
            mae_p = np.mean(np.abs(yt - preds["persistence"]))
            district = test.state_code.astype(str) + "_" + test.district_code.astype(str)
            for name, p in preds.items():
                mae = np.mean(np.abs(yt - p))
                lo, hi = (np.nan, np.nan) if name == "persistence" else _boot_skill(yt, p, preds["persistence"], district.values, n=300)
                rows.append(dict(target=t, horizon=h, model=name, MAE=mae, skill_vs_persistence=1 - mae / mae_p,
                                 skill_CI_low=lo, skill_CI_high=hi, n_test=len(test)))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(out, "forecast_results_extended.csv"), index=False)
    print("[Forecast v2]\n", r.round(4).to_string(index=False))
    return r


# ------------------------------------------------------------------ 3. cluster stability
def cluster_stability(df, tr, out, k_range=range(3, 9), n_boot=40):
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, adjusted_rand_score

    share_cols = [c for c in df if c.startswith("share_crop_")]
    level = ["irrigation_intensity", "cropping_intensity", "groundwater_share", "canal_share", "tank_share",
             "fallow_share", "forest_share", "sown_share", "crop_entropy"]
    recent = df[df.year_start >= 2015].groupby(KEYS)[level + share_cols].mean()
    trend = tr.set_index(KEYS)[["groundwater_share_slope", "irrigation_intensity_slope", "cropping_intensity_slope"]]
    prof = recent.join(trend).dropna(thresh=int(0.7 * (len(level) + len(share_cols) + 3)))
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(prof))
    rng = np.random.default_rng(SEED)
    rows = []
    for k in k_range:
        full = KMeans(k, n_init=10, random_state=SEED).fit(Z)
        aris = []
        for _ in range(n_boot):
            idx = rng.choice(len(Z), len(Z))
            km = KMeans(k, n_init=3, random_state=int(rng.integers(1e6))).fit(Z[idx])
            aris.append(adjusted_rand_score(full.labels_, km.predict(Z)))
        gmm = GaussianMixture(k, covariance_type="diag", n_init=3, random_state=SEED).fit(Z)
        rows.append(dict(k=k, silhouette=silhouette_score(Z, full.labels_), boot_ARI_mean=np.mean(aris),
                         boot_ARI_sd=np.std(aris), gmm_BIC=gmm.bic(Z),
                         kmeans_vs_gmm_ARI=adjusted_rand_score(full.labels_, gmm.predict(Z))))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(out, "cluster_stability.csv"), index=False)
    print("[Cluster stability]\n", s.round(3).to_string(index=False))
    return s


# ------------------------------------------------------------------ command-line / %run entry point
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run robust trends, extended forecasting and cluster stability")
    ap.add_argument("--data", default=".", help="folder with the 4 raw CSVs (only needed if no cached panel exists)")
    ap.add_argument("--out", default="study_outputs", help="output folder (same one the base pipeline used)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    panel_path = os.path.join(a.out, "clean_panel.csv")
    trend_path = os.path.join(a.out, "district_trends.csv")
    if os.path.exists(panel_path):
        print("Loading cached", panel_path)
        df = pd.read_csv(panel_path)
    else:
        print("No cached panel found -> rebuilding from raw CSVs (takes a few minutes)")
        import irrigation_pipeline as ip, study_pipeline as sp
        df = sp.clean_panel(ip.build_panel(a.data))
        df.to_csv(panel_path, index=False)
    if os.path.exists(trend_path):
        tr = pd.read_csv(trend_path)
    else:
        import study_pipeline as sp
        tr = sp.trend_analysis(df, a.out)

    trend_analysis_robust(df, a.out)
    cluster_stability(df, tr, a.out)
    forecast_with_baselines(df, a.out)
    print("\nDONE. Files written to:", os.path.abspath(a.out))
    for f in ["district_trends_robust.csv", "trend_robustness_summary.csv", "cluster_stability.csv",
              "forecast_results_extended.csv"]:
        print("  ", f, "OK" if os.path.exists(os.path.join(a.out, f)) else "MISSING")
