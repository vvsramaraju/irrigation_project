"""
Study: Forecasting, groundwater dependence and typology of irrigation across Indian districts (1998-2024)

Needs irrigation_pipeline.py in the same folder (reuses its panel builder / features).

  A. clean_panel()          proper cleaning BEFORE feature engineering
  B. trend_analysis()       Mann-Kendall style test + Theil-Sen slope per district
                            (groundwater share, irrigation intensity)
  C. typology()             KMeans clusters on recent profile + long-run trends
  D. multi_horizon_forecast()  1/3/5-year-ahead forecasts, skill vs persistence
  E. integrate()            forecast error by cluster; cluster x groundwater-trend table

Usage:  python study_pipeline.py --data <csv folder> --out study_outputs
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
import irrigation_pipeline as ip

warnings.filterwarnings("ignore")
SEED = 42
RAW_PREFIX = ("lu_", "irr_", "crop_", "cirr_")


# ---------------------------------------------------------------- A. cleaning
def clean_panel(panel):
    """Blank whole rows that fail basic sanity checks, then engineer features."""
    f = ip.add_features(panel)
    bad = (
        (f.cropping_intensity > 3.5) | (f.cropping_intensity < 0.9)
        | (f.fallow_share > 1) | (f.sown_share > 1.05)
        | (f.irrigation_intensity > 1.05) | (f.groundwater_share > 1.05) | (f.canal_share > 1.05)
    )
    print(f"[Clean] rows flagged as implausible: {int(bad.sum())} ({bad.mean():.1%})")
    raw = [c for c in panel.columns if c.startswith(RAW_PREFIX)]
    p = panel.copy()
    p.loc[bad.values, raw] = np.nan
    cleaned = ip.add_features(p)
    cleaned["flagged_bad"] = bad.values
    return cleaned


# ---------------------------------------------------------------- B. trends
def _trend(y, x, min_obs=15):
    m = ~np.isnan(y)
    if m.sum() < min_obs:
        return np.nan, np.nan, np.nan
    tau, p = stats.kendalltau(x[m], y[m])
    slope = stats.theilslopes(y[m], x[m])[0]
    return slope, tau, p


def trend_analysis(df, out, alpha=0.05):
    rows = []
    for (sc, dc), g in df.groupby(["state_code", "district_code"]):
        x = g.year_start.values.astype(float)
        r = dict(state_code=sc, district_code=dc, state_name=g.state_name.iloc[0], district_name=g.district_name.iloc[0])
        for v in ["groundwater_share", "irrigation_intensity", "cropping_intensity", "canal_share"]:
            s, t, p = _trend(g[v].values, x)
            r[f"{v}_slope"], r[f"{v}_tau"], r[f"{v}_p"] = s * 10, t, p   # slope per DECADE
        rows.append(r)
    tr = pd.DataFrame(rows)

    def cls(tau, p):
        return np.where(p.isna(), "insufficient data",
               np.where((p < alpha) & (tau > 0), "rising",
               np.where((p < alpha) & (tau < 0), "declining", "no significant trend")))
    tr["gw_trend_class"] = cls(tr.groundwater_share_tau, tr.groundwater_share_p)
    tr["irr_trend_class"] = cls(tr.irrigation_intensity_tau, tr.irrigation_intensity_p)
    tr.to_csv(os.path.join(out, "district_trends.csv"), index=False)

    print("[Trend] groundwater-share trend classes:\n", tr.gw_trend_class.value_counts().to_string())
    print("[Trend] irrigation-intensity trend classes:\n", tr.irr_trend_class.value_counts().to_string())
    st = (tr[tr.gw_trend_class != "insufficient data"].groupby("state_name")
            .agg(districts=("district_code", "count"),
                 pct_gw_rising=("gw_trend_class", lambda s: 100 * (s == "rising").mean()),
                 median_gw_slope_per_decade=("groundwater_share_slope", "median"))
            .sort_values("pct_gw_rising", ascending=False))
    st.to_csv(os.path.join(out, "state_groundwater_trends.csv"))
    return tr


# ---------------------------------------------------------------- C. typology
def typology(df, tr, out, k_range=range(3, 9)):
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    keys = ["state_code", "district_code"]
    share_cols = [c for c in df if c.startswith("share_crop_")]
    level = ["irrigation_intensity", "cropping_intensity", "groundwater_share", "canal_share", "tank_share",
             "fallow_share", "forest_share", "sown_share", "crop_entropy"]
    recent = df[df.year_start >= 2015].groupby(keys)[level + share_cols].mean()
    trend = tr.set_index(keys)[["groundwater_share_slope", "irrigation_intensity_slope", "cropping_intensity_slope"]]
    prof = recent.join(trend).dropna(thresh=int(0.7 * (len(level) + len(share_cols) + 3)))

    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(prof))
    sil = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)) for k in k_range}
    k = max(sil, key=sil.get)
    print("[Typology] silhouette:", {a: round(b, 3) for a, b in sil.items()}, "-> k =", k)
    prof["cluster"] = KMeans(k, n_init=30, random_state=SEED).fit_predict(Z)

    names = tr.groupby(keys)[["state_name", "district_name"]].first()
    prof = prof.join(names).reset_index()
    cols = level + ["groundwater_share_slope", "irrigation_intensity_slope"]
    summary = prof.groupby("cluster")[cols].mean().round(3)
    summary["n_districts"] = prof.groupby("cluster").size()
    summary["top_states"] = prof.groupby("cluster").state_name.agg(lambda s: ", ".join(s.value_counts().index[:3]))
    summary.to_csv(os.path.join(out, "cluster_summary.csv"))
    prof[["state_name", "district_name", "state_code", "district_code", "cluster"]].to_csv(
        os.path.join(out, "district_clusters.csv"), index=False)
    print(summary[["irrigation_intensity", "groundwater_share", "canal_share", "fallow_share", "n_districts", "top_states"]].to_string())

    xy = PCA(2, random_state=SEED).fit_transform(Z)
    plt.figure(figsize=(7, 6)); plt.scatter(xy[:, 0], xy[:, 1], c=prof.cluster, cmap="tab10", s=10)
    plt.title("District typology (PCA projection)"); plt.tight_layout()
    plt.savefig(os.path.join(out, "typology_pca.png"), dpi=150); plt.close()
    return prof[["state_code", "district_code", "cluster"]]


# ---------------------------------------------------------------- D. forecasting
def _features(df):
    cols = ip._feature_cols(df)
    return [c for c in cols if not c.startswith("y_")]


def multi_horizon_forecast(df, out, targets=("irrigation_intensity", "groundwater_share"), horizons=(1, 3, 5)):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score

    d = df.sort_values(["state_code", "district_code", "year_start"]).copy()
    g = d.groupby(["state_code", "district_code"])
    for t in targets:
        for h in horizons:
            d[f"y_{t}_h{h}"] = g[t].shift(-h)
    X_cols = _features(d)
    preds, rows = {}, []
    for t in targets:
        for h in horizons:
            y = f"y_{t}_h{h}"
            sub = d.dropna(subset=[y, t])
            tgt_year = sub.year_start + h
            train, test = sub[tgt_year <= 2017], sub[tgt_year >= 2021]    # no target-year overlap
            base = mean_absolute_error(test[y], test[t])
            res = {"persistence": test[t].values}
            models = {
                "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                       RidgeCV(alphas=np.logspace(-2, 3, 12))),
                "hist_gbm": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, l2_regularization=1.0,
                                                          early_stopping=True, random_state=SEED),
            }
            for name, m in models.items():
                m.fit(train[X_cols], train[y] - train[t])               # learn the change
                res[name] = m.predict(test[X_cols]) + test[t].values
            for name, p in res.items():
                mae = mean_absolute_error(test[y], p)
                rows.append(dict(target=t, horizon=h, model=name, MAE=mae, R2=r2_score(test[y], p),
                                 skill_vs_persistence=1 - mae / base, n_test=len(test)))
            preds[(t, h)] = test[["state_code", "district_code", "year_start", t, y]].assign(pred_gbm=res["hist_gbm"])
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(out, "forecast_results.csv"), index=False)
    print("[Forecast]\n", r.round(4).to_string(index=False))
    return r, preds


# ---------------------------------------------------------------- E. integration
def integrate(preds, clusters, tr, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = ["state_code", "district_code"]

    # forecast error by cluster (3-year horizon, irrigation intensity)
    p = preds[("irrigation_intensity", 3)].merge(clusters, on=keys)
    p["abs_err_gbm"] = (p.pred_gbm - p["y_irrigation_intensity_h3"]).abs()
    p["abs_err_persist"] = (p.irrigation_intensity - p["y_irrigation_intensity_h3"]).abs()
    byc = p.groupby("cluster")[["abs_err_persist", "abs_err_gbm"]].mean()
    byc["skill"] = 1 - byc.abs_err_gbm / byc.abs_err_persist
    byc.to_csv(os.path.join(out, "forecast_error_by_cluster.csv"))
    print("[Integrate] 3-yr forecast error by cluster:\n", byc.round(4).to_string())

    # cluster x groundwater-trend
    ct = pd.crosstab(clusters.merge(tr[keys + ["gw_trend_class"]], on=keys).cluster,
                     tr.set_index(keys).loc[pd.MultiIndex.from_frame(clusters[keys]), "gw_trend_class"].values,
                     normalize="index").round(3) * 100
    ct.to_csv(os.path.join(out, "cluster_vs_groundwater_trend_pct.csv"))
    print("[Integrate] % of districts per groundwater-trend class, by cluster:\n", ct.to_string())

    ax = ct.plot.bar(stacked=True, figsize=(8, 5), title="Groundwater-share trend class by cluster (%)")
    ax.set_ylabel("% of districts"); plt.tight_layout()
    plt.savefig(os.path.join(out, "cluster_vs_groundwater_trend.png"), dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="study_outputs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    panel = ip.build_panel(a.data)
    df = clean_panel(panel)
    df.to_csv(os.path.join(a.out, "clean_panel.csv"), index=False)
    tr = trend_analysis(df, a.out)
    clusters = typology(df, tr, a.out)
    results, preds = multi_horizon_forecast(df, a.out)
    integrate(preds, clusters, tr, a.out)
