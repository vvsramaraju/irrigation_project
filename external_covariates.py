"""
External covariates for the irrigation study: IMD gridded rainfall and CGWB groundwater levels.

Needs (Colab):  pip install imdlib geopandas shapely xarray
Files YOU must supply (not bundled):
  * a district boundary file (GeoJSON / shapefile) with state and district name columns
  * CGWB well-level groundwater data (e.g. from India-WRIS) as CSV with state, district, date, depth-to-water

Workflow
    import external_covariates as ec
    gdf   = ec.load_districts("districts.geojson", state_col="STATE", district_col="DISTRICT")
    rain  = ec.rainfall_for_districts(gdf, 1998, 2024, file_dir="imd_data")           # downloads via imdlib
    gw    = ec.groundwater_district_year("cgwb_levels.csv")                           # auto-detects columns
    xw    = ec.build_crosswalk(df, rain, gw, out)                                     # name matching -> REVIEW the CSV
    full  = ec.merge_covariates(df, rain, gw, xw)
    ec.forecast_with_covariates(full, out)

Agricultural-year convention (matches DES '1998-1999' -> year_start 1998):
    monsoon  = Jun-Sep of year_start        ag_year = Jun year_start .. May year_start+1
    gw_pre   = May of year_start (before kharif)   gw_post = Nov of year_start
"""
import os, re, difflib, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
KEYS = ["state_code", "district_code"]


# ---------------------------------------------------------------- name handling
_SYN = {"and": "", "the": "", "district": "", "dist": "", "of": "", "urban": "", "islands": "island"}


def norm_name(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return "".join(_SYN.get(w, w) for w in s.split())


def _best(name, candidates, cutoff):
    m = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    return (m[0], difflib.SequenceMatcher(None, name, m[0]).ratio()) if m else (None, 0.0)


def build_crosswalk(df, rain=None, gw=None, out=".", cutoff=0.75):
    """Match DES (state, district) names to external names. ALWAYS review the CSV it writes."""
    base = df.groupby(KEYS)[["state_name", "district_name"]].first().reset_index()
    base["s"] = base.state_name.map(norm_name)
    base["d"] = base.district_name.map(norm_name)
    rows = []
    for tag, ext in [("rain", rain), ("gw", gw)]:
        if ext is None:
            continue
        e = ext[["state_name", "district_name"]].drop_duplicates().copy()
        e["s"], e["d"] = e.state_name.map(norm_name), e.district_name.map(norm_name)
        states = e.s.unique().tolist()
        for _, r in base.iterrows():
            s_hit, s_sc = _best(r.s, states, cutoff)
            sub = e[e.s == s_hit] if s_hit else e
            d_hit, d_sc = _best(r.d, sub.d.unique().tolist(), cutoff)
            if not d_hit:                                   # state renamed/split (e.g. Telangana): search all states
                d_hit, d_sc = _best(r.d, e.d.unique().tolist(), 0.9)
                sub = e
            ext_row = sub[sub.d == d_hit].iloc[0] if d_hit else None
            rows.append(dict(source=tag, state_code=r.state_code, district_code=r.district_code,
                             des_state=r.state_name, des_district=r.district_name,
                             ext_state=None if ext_row is None else ext_row.state_name,
                             ext_district=None if ext_row is None else ext_row.district_name,
                             score=round(min(s_sc, d_sc) if d_hit else 0.0, 3)))
    xw = pd.DataFrame(rows)
    xw["needs_review"] = xw.score < 0.9
    path = os.path.join(out, "name_crosswalk_REVIEW.csv")
    xw.to_csv(path, index=False)
    print(f"[Crosswalk] {len(xw)} matches written to {path}; {int(xw.needs_review.sum())} need review "
          f"(unmatched or score < 0.9). Edit ext_state / ext_district there, then re-read it.")
    return xw


# ---------------------------------------------------------------- rainfall (IMD)
def load_districts(path, state_col, district_col):
    import geopandas as gpd
    g = gpd.read_file(path).to_crs(4326)
    g = g.rename(columns={state_col: "state_name", district_col: "district_name"})
    return g[["state_name", "district_name", "geometry"]].reset_index(drop=True)


def _district_masks(lat, lon, gdf):
    """Boolean mask (n_districts, n_lat, n_lon) of 0.25-degree cell centres inside each district."""
    import shapely
    LON, LAT = np.meshgrid(lon, lat)
    masks = np.zeros((len(gdf), len(lat), len(lon)), bool)
    for i, geom in enumerate(gdf.geometry.values):
        masks[i] = shapely.contains_xy(geom, LON, LAT)
        if not masks[i].any():                                     # tiny district: use nearest cell to centroid
            c = geom.centroid
            masks[i, np.abs(lat - c.y).argmin(), np.abs(lon - c.x).argmin()] = True
    return masks


def rainfall_from_xarray(ds, gdf, years, var="rain", wet_day_mm=2.5):
    """ds: xarray Dataset/DataArray with daily rainfall (time, lat, lon). Returns district-year table."""
    da = ds[var] if hasattr(ds, "data_vars") else ds
    da = da.where(da > -900)                                       # IMD no-data is -999
    lat, lon = da["lat"].values, da["lon"].values
    masks = _district_masks(lat, lon, gdf)
    rows = []
    for y in years:
        ag = da.sel(time=slice(f"{y}-06-01", f"{y + 1}-05-31"))
        mon = da.sel(time=slice(f"{y}-06-01", f"{y}-09-30"))
        if mon.sizes["time"] < 100 or ag.sizes["time"] < 360:      # skip incomplete agricultural years
            continue
        ag_tot = ag.sum("time", skipna=True).values
        mon_tot = mon.sum("time", skipna=True).values
        wet = (mon > wet_day_mm).sum("time").values.astype(float)
        post = da.sel(time=slice(f"{y}-10-01", f"{y}-12-31")).sum("time", skipna=True).values
        for i in range(len(gdf)):
            m = masks[i]
            rows.append(dict(state_name=gdf.state_name.iloc[i], district_name=gdf.district_name.iloc[i], year_start=y,
                             rain_monsoon=np.nanmean(mon_tot[m]), rain_agyear=np.nanmean(ag_tot[m]),
                             rain_postmonsoon=np.nanmean(post[m]), rain_wetdays=np.nanmean(wet[m])))
    return pd.DataFrame(rows)


def rainfall_for_districts(gdf, start_year, end_year, file_dir="imd_data"):
    """
    Download IMD 0.25-degree daily rainfall via imdlib (once for the whole range) and aggregate to districts.
    Resumable: finished years are cached in file_dir/rain_<year>.csv. Use end_year = last DES year_start (2023),
    because each agricultural year needs the following calendar year's file too.
    """
    import imdlib as imd
    os.makedirs(file_dir, exist_ok=True)
    imd.get_data("rain", start_year, end_year + 1, fn_format="yearwise", file_dir=file_dir)
    out = []
    for y in range(start_year, end_year + 1):
        cache = os.path.join(file_dir, f"rain_{y}.csv")
        if os.path.exists(cache):
            out.append(pd.read_csv(cache)); continue
        ds = imd.open_data("rain", y, y + 1, "yearwise", file_dir).get_xarray()
        r = rainfall_from_xarray(ds, gdf, [y])
        r.to_csv(cache, index=False)
        out.append(r)
        print("rain", y, "done", flush=True)
    r = pd.concat(out, ignore_index=True)
    r.to_csv(os.path.join(file_dir, "district_rainfall.csv"), index=False)
    return r


# ---------------------------------------------------------------- groundwater levels (CGWB)
def _find(cols, *keys):
    low = {c.lower(): c for c in cols}
    for k in keys:
        for lc, c in low.items():
            if k in lc:
                return c
    return None


def inspect_file(path, n=5):
    d = pd.read_csv(path, nrows=n)
    print(d.dtypes.to_string()); print(d.head(n).to_string())


def groundwater_district_year(path, state_col=None, district_col=None, date_col=None, depth_col=None,
                              min_wells=5, pre_months=(4, 5, 6), post_months=(10, 11, 12), max_depth=200):
    """
    Well-level readings -> district-year (m below ground level; LARGER = DEEPER = worse).
    gw_pre  = median Apr-Jun depth of year_start (pre-monsoon), gw_post = median Oct-Dec depth (post-monsoon),
    gw_recovery = gw_pre - gw_post (positive = water table rose over the monsoon).
    Each well is first reduced to its own seasonal median, then the district median is taken across wells;
    a district-season is NaN if fewer than `min_wells` wells contribute. Keeps state_code / district_code when present.
    """
    d = pd.read_csv(path, low_memory=False)
    sc = state_col or _find(d.columns, "state_name", "state")
    dc = district_col or _find(d.columns, "district_name", "district")
    tc = date_col or _find(d.columns, "date", "time", "month", "year")
    vc = depth_col or _find(d.columns, "currentlevel", "depth", "dtwl", "gwl", "level")
    well = _find(d.columns, "station_name", "station", "well", "site", "code")
    scode = "state_code" if "state_code" in d.columns else None
    dcode = "district_code" if "district_code" in d.columns else None
    print(f"[GW] columns: state={sc}, district={dc}, date={tc}, depth={vc}, well={well}, codes={scode},{dcode}")
    if None in (sc, dc, tc, vc):
        raise ValueError("Could not detect all columns; pass state_col/district_col/date_col/depth_col explicitly")
    d["_t"] = pd.to_datetime(d[tc], errors="coerce")
    d["_v"] = pd.to_numeric(d[vc], errors="coerce")
    d = d.dropna(subset=["_t", "_v"])
    d = d[(d._v > 0) & (d._v < max_depth)]                         # drop zero / impossible depths
    d["year_start"], d["month"] = d._t.dt.year, d._t.dt.month
    grp = [c for c in [sc, dc, scode, dcode] if c] + ["year_start"]

    def season(months, name):
        s_ = d[d.month.isin(months)]
        a = s_.groupby(grp + ([well] if well else []))["_v"].median().reset_index()
        g = a.groupby(grp)["_v"].agg(["median", "count"]).reset_index()
        g.loc[g["count"] < min_wells, "median"] = np.nan
        return g.rename(columns={"median": name, "count": f"n_wells_{name}"})

    pre, post = season(pre_months, "gw_pre"), season(post_months, "gw_post")
    g = pre.merge(post, on=grp, how="outer").rename(columns={sc: "state_name", dc: "district_name"})
    g["gw_recovery"] = g.gw_pre - g.gw_post
    print(f"[GW] district-years: {len(g)} | with gw_pre: {g.gw_pre.notna().sum()} | with gw_post: {g.gw_post.notna().sum()} "
          f"| years {int(g.year_start.min())}-{int(g.year_start.max())}")
    return g


def groundwater_validation(df, gw, out, trends=None, clusters=None, min_years=7, alpha=0.10):
    """
    Validate the DES groundwater-SHARE indicator against physical water-table depth.
    Per district: Theil-Sen slope of post/pre-monsoon depth over the years available (m/yr, positive = deepening).
    Compared with (a) the 1998-2023 groundwater-share trend and (b) the change in groundwater share between
    the first and second half of the overlap window. Spearman correlations and a class cross-tab are written to `out`.
    """
    from scipy import stats
    rows = []
    for (sc_, dc_), x in gw.groupby(KEYS):
        r = dict(state_code=sc_, district_code=dc_, state_name=x.state_name.iloc[0], district_name=x.district_name.iloc[0])
        for v in ["gw_post", "gw_pre"]:
            xx = x.dropna(subset=[v])
            r[f"{v}_years"] = len(xx)
            r[f"{v}_mean"] = xx[v].mean() if len(xx) else np.nan
            if len(xx) >= min_years and xx.year_start.nunique() >= min_years:
                r[f"{v}_slope_m_per_yr"] = stats.theilslopes(xx[v].values, xx.year_start.values.astype(float))[0]
                r[f"{v}_p"] = stats.kendalltau(xx.year_start.values, xx[v].values)[1]
            else:
                r[f"{v}_slope_m_per_yr"] = r[f"{v}_p"] = np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    t["water_table_class"] = np.where(t.gw_post_p.isna(), "insufficient data",
                              np.where((t.gw_post_p < alpha) & (t.gw_post_slope_m_per_yr > 0), "deepening",
                              np.where((t.gw_post_p < alpha) & (t.gw_post_slope_m_per_yr < 0), "recovering", "no significant trend")))
    lo, hi = int(gw.year_start.min()), int(gw.year_start.max())
    mid = (lo + hi) // 2
    share = df.groupby(KEYS + ["year_start"])["groundwater_share"].mean().reset_index()
    early = share[share.year_start.between(lo, mid)].groupby(KEYS).groundwater_share.mean()
    late = share[share.year_start.between(mid + 1, min(hi, 2023))].groupby(KEYS).groundwater_share.mean()
    t = t.merge((late - early).rename("delta_gw_share_overlap").reset_index(), on=KEYS, how="left")
    if trends is not None:
        keep = [c for c in ["groundwater_share_slope", "gw_trend_class", "irrigation_intensity_slope", "irr_trend_class"] if c in trends]
        t = t.merge(trends[KEYS + keep], on=KEYS, how="left")
    if clusters is not None:
        t = t.merge(clusters[KEYS + ["cluster"]], on=KEYS, how="left")
    t.to_csv(os.path.join(out, "groundwater_validation_districts.csv"), index=False)

    print(f"[GW validation] overlap window {lo}-{hi}; districts with a usable depth trend: {int(t.gw_post_slope_m_per_yr.notna().sum())}")
    print(t.water_table_class.value_counts().to_string())
    res = []
    for a, b in [("gw_post_slope_m_per_yr", "groundwater_share_slope"), ("gw_post_slope_m_per_yr", "delta_gw_share_overlap"),
                 ("gw_post_slope_m_per_yr", "irrigation_intensity_slope"), ("gw_pre_slope_m_per_yr", "groundwater_share_slope")]:
        if a in t and b in t:
            m = t[[a, b]].dropna()
            if len(m) > 10:
                rho, p = stats.spearmanr(m[a], m[b])
                res.append(dict(depth_trend=a, des_indicator=b, n=len(m), spearman_rho=rho, p_value=p))
    cr = pd.DataFrame(res)
    cr.to_csv(os.path.join(out, "groundwater_validation_correlations.csv"), index=False)
    print(cr.round(3).to_string(index=False))
    if "gw_trend_class" in t:
        ct = pd.crosstab(t.gw_trend_class, t.water_table_class)
        ct.to_csv(os.path.join(out, "groundwater_validation_crosstab.csv"))
        print(ct.to_string())
    return t, cr


# ---------------------------------------------------------------- merge + features
def merge_covariates(df, rain=None, gw=None, xw=None):
    """Attach external covariates to the DES panel via the (reviewed) crosswalk, then add anomalies and lags."""
    out = df.copy()
    for tag, ext in [("rain", rain), ("gw", gw)]:
        if ext is None:
            continue
        if set(KEYS).issubset(ext.columns):                          # codes available: exact join, no name matching
            j = ext.drop(columns=[c for c in ["state_name", "district_name"] if c in ext.columns])
            j = j.groupby(KEYS + ["year_start"], as_index=False).first()
        else:
            m = xw[xw.source == tag].dropna(subset=["ext_state", "ext_district"])
            e = ext.rename(columns={"state_name": "ext_state", "district_name": "ext_district"})
            j = m[KEYS + ["ext_state", "ext_district"]].merge(e, on=["ext_state", "ext_district"])
            j = j.drop(columns=["ext_state", "ext_district"])
        out = out.merge(j, on=KEYS + ["year_start"], how="left")
    out = out.sort_values(KEYS + ["year_start"]).reset_index(drop=True)
    g = out.groupby(KEYS)
    for c in [c for c in ["rain_monsoon", "rain_agyear", "gw_pre", "gw_post", "gw_recovery"] if c in out]:
        # anomaly vs the district's own PAST (expanding mean/sd shifted by 1): no look-ahead
        mu = g[c].transform(lambda s: s.shift(1).expanding(5).mean())
        sd = g[c].transform(lambda s: s.shift(1).expanding(5).std())
        out[f"{c}_anom"] = (out[c] - mu) / sd.replace(0, np.nan)
        for L in (1, 2):
            out[f"{c}_lag{L}"] = g[c].shift(L)
    if "rain_monsoon_anom" in out:                                  # drought indicator
        out["monsoon_deficit"] = (out.rain_monsoon_anom < -1).astype(float)
    return out


def fill_missing_with_state_mean(full, cols=("rain_monsoon", "rain_agyear", "rain_postmonsoon", "rain_wetdays")):
    """
    Districts created after 2011 have no polygon in the 2011 boundary file, so they get no rainfall.
    Approximation: fill them with the mean of the matched districts of the SAME state and year, and flag the rows
    (rain_state_filled = 1) so results can be re-run without them as a robustness check.
    """
    out = full.copy()
    cols = [c for c in cols if c in out.columns]
    first = cols[0]
    out["rain_state_filled"] = out[first].isna().astype(int)
    for c in cols:
        m = out.groupby(["state_code", "year_start"])[c].transform("mean")
        out[c] = out[c].fillna(m)
    still = out[first].isna() & out.year_start.between(1998, 2023)
    print(f"[Rain fill] filled {int(out.rain_state_filled.sum())} district-years with state means; "
          f"{int(still.sum())} still missing (no matched district in that state-year)")
    return out


# ---------------------------------------------------------------- forecasting with covariates
def forecast_with_covariates(df, out, target="irrigation_intensity", horizons=(1, 3, 5), oracle=True):
    """
    Compares: persistence | state_mean_change | gbm_base | gbm_plus_covariates (information available at origin)
    and, if oracle=True, gbm_oracle_rain which also sees mean monsoon-rainfall anomaly over the next h years
    (NOT a real forecast: it answers 'does rainfall explain change?').
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    import irrigation_pipeline as ip
    import study_extensions as se

    d = df.sort_values(KEYS + ["year_start"]).copy()
    g = d.groupby(KEYS)
    cov = [c for c in d.columns if c.startswith(("rain_", "gw_", "monsoon_deficit"))]
    d["slope8"] = se._own_slope(d, target)
    for h in horizons:
        d[f"y_h{h}"] = g[target].shift(-h)
        if "rain_monsoon_anom" in d:
            d[f"future_rain_anom_h{h}"] = g["rain_monsoon_anom"].transform(
                lambda s, h=h: sum(s.shift(-k) for k in range(1, h + 1)) / h)
    base = [c for c in ip._feature_cols(d) if not c.startswith(("y_", "slope8", "future_", "rain_", "gw_", "monsoon_deficit"))]
    rows = []
    for h in horizons:
        y = f"y_h{h}"
        sub = d.dropna(subset=[y, target])
        ty = sub.year_start + h
        train, test = sub[ty <= 2017], sub[ty >= 2021]
        chg = (train[y] - train[target]).groupby(train.state_code).mean()
        preds = {"persistence": test[target].values,
                 "state_mean_change": (test[target] + test.state_code.map(chg).fillna(0)).values}
        sets = {"gbm_base": base + ["slope8"], "gbm_plus_covariates": base + ["slope8"] + cov}
        if oracle and f"future_rain_anom_h{h}" in d:
            sets["gbm_oracle_rain"] = sets["gbm_plus_covariates"] + [f"future_rain_anom_h{h}"]
        for name, cols in sets.items():
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, l2_regularization=1.0,
                                              early_stopping=True, random_state=SEED)
            cols = list(dict.fromkeys(cols))
            m.fit(train[cols], train[y] - train[target])
            preds[name] = m.predict(test[cols]) + test[target].values
        yt = test[y].values
        mae_p = np.mean(np.abs(yt - preds["persistence"]))
        dist = (test.state_code.astype(str) + "_" + test.district_code.astype(str)).values
        for name, p in preds.items():
            mae = np.mean(np.abs(yt - p))
            lo, hi = (np.nan, np.nan) if name == "persistence" else se._boot_skill(yt, p, preds["persistence"], dist, n=300)
            rows.append(dict(target=target, horizon=h, model=name, MAE=mae, skill_vs_persistence=1 - mae / mae_p,
                             skill_CI_low=lo, skill_CI_high=hi, n_test=len(test)))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(out, "forecast_results_with_covariates.csv"), index=False)
    print("[Forecast + covariates]\n", r.round(4).to_string(index=False))
    return r


# ---------------------------------------------------------------- validation of the trend atlas against water levels
def validate_trends_with_levels(full, tr, out, clusters=None, years=(2013, 2019), min_years=5):
    """
    Does a rising groundwater SHARE (area statistics) go with a DEEPENING water table (CGWB levels)?
    CGWB pre-monsoon levels exist only for ~2013-2019, so this is a short-window validation, not a long trend.
      gw_depth_slope  : Theil-Sen slope of pre-monsoon depth (m/year; positive = water table getting deeper)
      share_slope_win : Theil-Sen slope of groundwater share over the same window (per year)
      gw_share_slope  : full-period slope per decade from the trend atlas
    """
    from scipy import stats
    w = full[full.year_start.between(*years)]
    rows = []
    for (sc, dc), g in w.groupby(KEYS):
        x = g.year_start.values.astype(float)
        r = dict(state_code=sc, district_code=dc)
        m = g.gw_pre.notna().values
        if m.sum() >= min_years:
            r["gw_depth_slope"] = stats.theilslopes(g.gw_pre.values[m], x[m])[0]
            r["gw_depth_tau"], r["gw_depth_p"] = stats.kendalltau(x[m], g.gw_pre.values[m])
            r["n_years_depth"] = int(m.sum())
        m2 = g.groundwater_share.notna().values
        if m2.sum() >= min_years:
            r["share_slope_win"] = stats.theilslopes(g.groundwater_share.values[m2], x[m2])[0]
        rows.append(r)
    v = pd.DataFrame(rows).dropna(subset=["gw_depth_slope"])
    keep = [c for c in ["state_code", "district_code", "groundwater_share_slope", "gw_trend_class",
                        "irrigation_intensity_slope", "state_name", "district_name"] if c in tr.columns]
    v = v.merge(tr[keep], on=KEYS, how="left")
    if clusters is not None:
        v = v.merge(clusters[KEYS + ["cluster"]], on=KEYS, how="left")
    v.to_csv(os.path.join(out, "validation_levels_vs_trends.csv"), index=False)

    print(f"[Validation] districts with >= {min_years} years of pre-monsoon levels: {len(v)}")
    for a, b in [("gw_depth_slope", "groundwater_share_slope"), ("gw_depth_slope", "share_slope_win"),
                 ("gw_depth_slope", "irrigation_intensity_slope")]:
        s = v[[a, b]].dropna()
        if len(s) > 10:
            rho, p = stats.spearmanr(s[a], s[b])
            print(f"  Spearman({a}, {b}): rho={rho:+.3f}, p={p:.3g}, n={len(s)}")
    if "gw_trend_class" in v:
        print("  mean water-table deepening (m/yr) by groundwater-share trend class:\n",
              v.groupby("gw_trend_class").gw_depth_slope.agg(["mean", "median", "count"]).round(3).to_string())
    if "cluster" in v:
        print("  by district type:\n", v.groupby("cluster").gw_depth_slope.agg(["mean", "median", "count"]).round(3).to_string())
    return v
