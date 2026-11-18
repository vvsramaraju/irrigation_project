"""
District-level Land Use & Irrigation analytics pipeline (India, 1998-99 to 2023-24)

Datasets (Land Use Statistics, DES, Ministry of Agriculture):
  classification-of-area.csv, sources-of-irrigation.csv,
  area-under-crops.csv, crop-wise-irrigated-area.csv

Stages
  1. build_panel()       -> clean district x year panel (one row per district-year)
  2. add_features()      -> shares, lags, rolling stats, growth rates
  3. task_ml_forecast()  -> Gradient boosting: predict NEXT-year irrigation intensity
  4. task_dl_forecast()  -> GRU sequence model on the same target (needs PyTorch)
  5. task_clustering()   -> KMeans (+ optional autoencoder) district typologies
  6. task_anomaly()      -> IsolationForest to flag suspicious / abrupt district-years

Run:  python irrigation_pipeline.py --data /path/to/csvs --out outputs
Deps: pandas numpy scikit-learn matplotlib   (optional: torch, shap, lightgbm)
"""
import argparse, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
KEY = ["state_code", "district_code", "year"]
SEED = 42


# ----------------------------------------------------------------------------
# 1. LOAD + CLEAN
# ----------------------------------------------------------------------------
def _read(path):
    df = pd.read_csv(path)
    df["year_start"] = df["year"].str[:4].astype(int)      # '1998-1999' -> 1998
    return df.dropna(subset=["district_code"])


def _wide(df, cols, value, prefix):
    """Pivot long -> wide. min_count=1 keeps NaN (= not reported) instead of 0."""
    d = df.copy()
    d["_k"] = d[cols].astype(str).agg("|".join, axis=1) if len(cols) > 1 else d[cols[0]].astype(str)
    w = d.pivot_table(index=KEY + ["state_name", "district_name", "year_start"],
                      columns="_k", values=value, aggfunc="sum", observed=True)
    w.columns = [f"{prefix}{c}".replace(" ", "_").lower() for c in w.columns]
    return w.reset_index()


def build_panel(data_dir):
    p = lambda f: os.path.join(data_dir, f)

    # --- land use (Classification of Area) ---------------------------------
    la = _read(p("classification-of-area.csv"))
    la["label"] = la["land_use_metrics"].fillna(la["area_sub_classification"])
    land = _wide(la, ["label"], "area", "lu_")

    # --- irrigation sources -------------------------------------------------
    ir = _read(p("sources-of-irrigation.csv"))
    ir["label"] = ir["irrigated_area_type"].str.split().str[0] + "_" + ir["area_sub_classification"]
    src = _wide(ir, ["label"], "irrigated_area", "irr_")

    # --- crop areas: collapse to crop-TYPE level (16 types, not 69 crops) ----
    ac = _read(p("area-under-crops.csv"))
    ac = ac.dropna(subset=["crop_type"])
    ac = ac.groupby(KEY + ["state_name", "district_name", "year_start", "crop_type"],
                    as_index=False)["area"].sum(min_count=1)
    crops = _wide(ac, ["crop_type"], "area", "crop_")

    # --- crop-wise irrigated area (by crop type) -----------------------------
    ci = _read(p("crop-wise-irrigated-area.csv")).dropna(subset=["crop_type"])
    ci = ci.groupby(KEY + ["state_name", "district_name", "year_start", "crop_type"],
                    as_index=False)["area"].sum(min_count=1)
    cirr = _wide(ci, ["crop_type"], "area", "cirr_")

    on = KEY + ["state_name", "district_name", "year_start"]
    panel = land.merge(src, on=on, how="outer").merge(crops, on=on, how="left").merge(cirr, on=on, how="left")
    return panel.sort_values(["state_code", "district_code", "year_start"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. FEATURES
# ----------------------------------------------------------------------------
def add_features(df):
    df = df.copy()
    g = lambda c: df[c] if c in df else np.nan
    nsa = g("lu_net_area_sown").where(lambda s: s > 0)
    rep = g("lu_reporting_area").where(lambda s: s > 0)

    irr_net_cols = [c for c in df if c.startswith("irr_net_")]
    irr_gross_cols = [c for c in df if c.startswith("irr_gross_")]
    df["net_irrigated"] = df[irr_net_cols].sum(axis=1, min_count=len(irr_net_cols) - 1)
    df["gross_irrigated"] = df[irr_gross_cols].sum(axis=1, min_count=len(irr_gross_cols) - 1)

    # target + engineered ratios
    df["irrigation_intensity"] = (df["net_irrigated"] / nsa).clip(0, 1.2)   # share of sown area irrigated
    df["cropping_intensity"] = g("lu_cropped_area") / nsa                    # gross cropped / net sown
    df["sown_share"] = nsa / rep
    df["forest_share"] = g("lu_forests") / rep
    df["fallow_share"] = (g("lu_current_fallow") + g("lu_fallow_lands_other_than_current_fallows")) / rep
    df["multi_crop_share"] = g("lu_area_sown_more_than_once") / nsa
    df["groundwater_share"] = (g("irr_net_tubewell") + g("irr_net_other_well")) / df["net_irrigated"].where(lambda s: s > 0)
    df["canal_share"] = (g("irr_net_government") + g("irr_net_private")) / df["net_irrigated"].where(lambda s: s > 0)
    df["tank_share"] = g("irr_net_tank") / df["net_irrigated"].where(lambda s: s > 0)

    crop_cols = [c for c in df if c.startswith("crop_")]
    tot = df[crop_cols].sum(axis=1, min_count=1)
    for c in crop_cols:
        df[f"share_{c}"] = df[c] / tot.where(tot > 0)
    # crop diversity (Shannon entropy over crop-type shares)
    sh = df[[f"share_{c}" for c in crop_cols]].fillna(0).clip(1e-9, 1)
    df["crop_entropy"] = -(sh * np.log(sh)).sum(axis=1).where(tot.notna())

    # time-series features (per district)
    grp = df.groupby(["state_code", "district_code"])
    base = ["irrigation_intensity", "cropping_intensity", "groundwater_share", "canal_share",
            "tank_share", "fallow_share", "multi_crop_share", "crop_entropy", "sown_share"]
    for c in base:
        for L in (1, 2, 3):
            df[f"{c}_lag{L}"] = grp[c].shift(L)
        df[f"{c}_roll3"] = grp[c].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
    df["irr_growth_1y"] = grp["irrigation_intensity"].diff(1)
    df["target_next"] = grp["irrigation_intensity"].shift(-1)                # predict next year
    # Impute only the *share* columns; raw areas stay NaN.
    return df


# ----------------------------------------------------------------------------
# 3. ML: gradient boosting with a TEMPORAL split
# ----------------------------------------------------------------------------
def _feature_cols(df):
    drop = {"target_next", "year", "state_name", "district_name", "year_start"}
    return [c for c in df.columns if c not in drop and df[c].dtype != object
            and not c.startswith(("lu_", "irr_", "crop_", "cirr_"))
            and c not in {"net_irrigated", "gross_irrigated", "state_code", "district_code"}] \
        + ["state_code"]


def task_ml_forecast(df, out):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    d = df.dropna(subset=["target_next", "irrigation_intensity"])
    X_cols = _feature_cols(d)
    train, val, test = d[d.year_start <= 2017], d[d.year_start.between(2018, 2020)], d[d.year_start >= 2021]
    print(f"[ML] rows train/val/test = {len(train)}/{len(val)}/{len(test)}  features={len(X_cols)}")

    models = {
        "persistence(naive)": None,
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=SEED)),
        "hist_gbm": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                                  l2_regularization=1.0, early_stopping=True,
                                                  validation_fraction=0.1, random_state=SEED),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        if m is None:
            pred = test["irrigation_intensity"].values
        else:
            # model the CHANGE vs this year (beats naive persistence), then add it back
            m.fit(train[X_cols], train["target_next"] - train["irrigation_intensity"]); fitted[name] = m
            pred = m.predict(test[X_cols]) + test["irrigation_intensity"].values
        rows.append(dict(model=name, MAE=mean_absolute_error(test.target_next, pred),
                         R2=r2_score(test.target_next, pred)))
    res = pd.DataFrame(rows).sort_values("MAE"); print(res.to_string(index=False))
    res.to_csv(os.path.join(out, "ml_results.csv"), index=False)

    # permutation importance on the best tree model
    from sklearn.inspection import permutation_importance
    best = fitted["hist_gbm"]
    sample = test.sample(min(2000, len(test)), random_state=SEED)
    pi = permutation_importance(best, sample[X_cols], sample["target_next"] - sample["irrigation_intensity"],
                                n_repeats=5, random_state=SEED)
    imp = pd.Series(pi.importances_mean, X_cols).sort_values(ascending=False).head(15)
    imp[::-1].plot.barh(figsize=(7, 5), title="Permutation importance (HistGBM)")
    plt.tight_layout(); plt.savefig(os.path.join(out, "ml_importance.png"), dpi=150); plt.close()
    return res


# ----------------------------------------------------------------------------
# 4. DL: GRU over 5-year windows -> next-year irrigation intensity
# ----------------------------------------------------------------------------
def task_dl_forecast(df, out, window=5, epochs=60):
    try:
        import torch, torch.nn as nn
    except Exception:
        print("[DL] PyTorch not available -> `pip install torch` to run this step"); return
    from sklearn.metrics import mean_absolute_error, r2_score

    torch.manual_seed(SEED); np.random.seed(SEED)
    feats = ["irrigation_intensity", "cropping_intensity", "groundwater_share", "canal_share",
             "tank_share", "fallow_share", "multi_crop_share", "crop_entropy", "sown_share"]
    d = df.sort_values(["state_code", "district_code", "year_start"])
    # fill gaps within district (ffill/bfill), remaining NaN -> column median
    d[feats] = d.groupby(["state_code", "district_code"])[feats].transform(lambda s: s.ffill().bfill())
    d[feats] = d[feats].fillna(d[feats].median())

    mu, sd = d[d.year_start <= 2017][feats].mean(), d[d.year_start <= 2017][feats].std().replace(0, 1)
    d[feats] = (d[feats] - mu) / sd

    Xs, ys, yrs = [], [], []
    for _, g in d.groupby(["state_code", "district_code"]):
        a, y, t = g[feats].values, g["irrigation_intensity"].values, g["year_start"].values
        orig = df.loc[g.index, "irrigation_intensity"].values
        for i in range(len(g) - window):
            if np.isnan(orig[i + window]): continue
            Xs.append(a[i:i + window]); ys.append(y[i + window]); yrs.append(t[i + window])
    X, y, yrs = np.array(Xs, dtype="float32"), np.array(ys, dtype="float32"), np.array(yrs)
    # target is standardised too -> invert at eval time
    m_t, s_t = mu["irrigation_intensity"], sd["irrigation_intensity"]
    tr, va, te = yrs <= 2017, (yrs >= 2018) & (yrs <= 2020), yrs >= 2021
    print(f"[DL] sequences train/val/test = {tr.sum()}/{va.sum()}/{te.sum()}")

    class GRUNet(nn.Module):
        def __init__(s, n_in, h=64):
            super().__init__()
            s.gru = nn.GRU(n_in, h, num_layers=2, batch_first=True, dropout=0.2)
            s.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(s, x): return s.head(s.gru(x)[0][:, -1]).squeeze(-1)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = GRUNet(len(feats)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    T = lambda a: torch.tensor(a).to(dev)
    Xtr, ytr, Xva, yva, Xte = T(X[tr]), T(y[tr]), T(X[va]), T(y[va]), T(X[te])
    best, best_state, patience = 1e9, None, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]; opt.zero_grad()
            nn.functional.huber_loss(net(Xtr[b]), ytr[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = nn.functional.l1_loss(net(Xva), yva).item()
        if v < best: best, best_state, patience = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad(): pred = net(Xte).cpu().numpy() * s_t + m_t
    true = y[te] * s_t + m_t
    print(f"[DL] GRU test MAE={mean_absolute_error(true, pred):.4f}  R2={r2_score(true, pred):.3f}")
    pd.DataFrame({"true": true, "pred": pred}).to_csv(os.path.join(out, "dl_predictions.csv"), index=False)


# ----------------------------------------------------------------------------
# 5. CLUSTERING: district typologies
# ----------------------------------------------------------------------------
def task_clustering(df, out, k_range=range(3, 9)):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    recent = df[df.year_start >= 2015]
    share_cols = [c for c in df if c.startswith("share_crop_")]
    feats = ["irrigation_intensity", "cropping_intensity", "groundwater_share", "canal_share", "tank_share",
             "fallow_share", "forest_share", "sown_share", "crop_entropy"] + share_cols
    prof = recent.groupby(["state_name", "district_name"])[feats].mean()
    prof = prof.dropna(thresh=int(0.6 * len(feats)))
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(prof))

    scores = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)) for k in k_range}
    k = max(scores, key=scores.get); print(f"[Cluster] silhouette by k: {scores} -> k={k}")
    prof["cluster"] = KMeans(k, n_init=20, random_state=SEED).fit_predict(Z)
    prof.groupby("cluster")[feats[:9]].mean().round(2).to_csv(os.path.join(out, "cluster_profiles.csv"))
    prof.reset_index()[["state_name", "district_name", "cluster"]].to_csv(os.path.join(out, "district_clusters.csv"), index=False)

    xy = PCA(2, random_state=SEED).fit_transform(Z)
    plt.figure(figsize=(7, 6)); plt.scatter(xy[:, 0], xy[:, 1], c=prof.cluster, cmap="tab10", s=10)
    plt.title("District typologies (PCA of land-use / irrigation / crop-mix)"); plt.tight_layout()
    plt.savefig(os.path.join(out, "clusters_pca.png"), dpi=150); plt.close()


# ----------------------------------------------------------------------------
# 6. ANOMALY DETECTION: abrupt jumps / data-entry errors
# ----------------------------------------------------------------------------
def task_anomaly(df, out):
    from sklearn.ensemble import IsolationForest
    cols = ["irrigation_intensity", "cropping_intensity", "sown_share", "fallow_share",
            "groundwater_share", "irr_growth_1y"]
    d = df.dropna(subset=cols).copy()
    d["anomaly_score"] = -IsolationForest(n_estimators=300, contamination=0.01, random_state=SEED).fit(d[cols]).score_samples(d[cols])
    top = d.sort_values("anomaly_score", ascending=False).head(200)
    top[["state_name", "district_name", "year", "anomaly_score"] + cols].to_csv(os.path.join(out, "anomalies.csv"), index=False)
    print("[Anomaly] top-200 written to anomalies.csv")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/user-data/uploads")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    panel = build_panel(a.data)
    print("panel:", panel.shape, "| districts:", panel.groupby(["state_code", "district_code"]).ngroups)
    feat = add_features(panel)
    feat.to_csv(os.path.join(a.out, "district_year_panel.csv"), index=False)

    task_ml_forecast(feat, a.out)
    task_dl_forecast(feat, a.out)
    task_clustering(feat, a.out)
    task_anomaly(feat, a.out)
