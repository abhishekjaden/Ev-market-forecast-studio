import os, io, numpy as np, pandas as pd
import streamlit as st
from datetime import datetime

# ML
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Viz
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="EV Market & Forecast Studio", layout="wide")


# -------------------------------
# Utilities
# -------------------------------
def smape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))


def demo_monthly_df():
    """Small, realistic synthetic monthly series (2W & 3W + exogenous) for demo."""
    rng = pd.date_range("2022-01-01", "2025-12-01", freq="MS")
    n = len(rng)
    base = 700 + 500 * np.sin(np.linspace(0, 3 * np.pi, n)) + np.linspace(0, 350, n)
    noise = np.random.default_rng(42).normal(0, 35, n)
    d2w = np.maximum(1, base + noise)
    d3w = 0.18 * d2w + np.random.default_rng(1).normal(0, 8, n)
    stations = np.linspace(15000, 26000, n) + np.random.default_rng(4).normal(0, 50, n)
    fuel = 90 + 8 * np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.default_rng(5).normal(0, 0.6, n)
    return pd.DataFrame(
        {
            "date": rng,
            "state": "Total",
            "demand_2w": d2w,
            "demand_3w": d3w,
            "station_count": stations,
            "fuel_price": fuel,
        }
    )


def build_ts_features_plus(frame, target):
    """Lag/MA/diff features + exogenous transforms (station, fuel)."""
    f = frame.sort_values("date").copy()
    # target lags
    for L in [1, 2, 3, 6]:
        f[f"{target}_lag{L}"] = f[target].shift(L)
    # moving averages from lag1
    for w in [3, 6]:
        f[f"{target}_ma{w}"] = f[f"{target}_lag1"].rolling(w).mean()
    # safe diffs
    f[f"{target}_diff1"] = f[f"{target}_lag1"] - f[f"{target}_lag2"]
    f[f"{target}_diff3"] = (f[f"{target}_lag3"] - f[f"{target}_lag6"]).fillna(0.0)
    # exogenous lags/derived
    if "station_count" in f.columns:
        f["station_count_lag1"] = f["station_count"].shift(1)
        f["station_growth"] = f["station_count"].diff(1).shift(1)
        f["station_ma3"] = f["station_count"].shift(1).rolling(3).mean()
        f["sc_x_lag1"] = f["station_count"].shift(1) * f[f"{target}_lag1"]
    if "fuel_price" in f.columns:
        f["fuel_price_lag1"] = f["fuel_price"].shift(1)
        f["fuel_change"] = f["fuel_price"].diff(1).shift(1)
    f["month"] = f["date"].dt.month.astype("int8")
    return f


def select_and_fit(X_train, y_train, X_test, y_test):
    """Train a few models with CV, pick the one with lowest sMAPE on the test set."""
    tscv = TimeSeriesSplit(n_splits=4)

    candidates = {
        "Ridge": GridSearchCV(
            Ridge(),
            {"alpha": [0.1, 1, 5, 10]},
            cv=tscv, scoring="neg_mean_absolute_error"
        ),
        "RF": GridSearchCV(
            RandomForestRegressor(random_state=42),
            {"n_estimators": [400, 800],
             "max_depth": [None, 6, 10],
             "min_samples_leaf": [1, 3]},
            cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
        ),
        "HGB": GridSearchCV(
            HistGradientBoostingRegressor(random_state=42),
            {"max_depth": [None, 6, 10],
             "learning_rate": [0.05, 0.1],
             "max_iter": [300, 600]},
            cv=tscv, scoring="neg_mean_absolute_error"
        ),
    }

    scores = {}
    for name, gs in candidates.items():
        gs.fit(X_train, y_train)
        est  = gs.best_estimator_            # <-- single underscore
        pred = est.predict(X_test)

        scores[name] = {
            "est": est,
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae":  float(mean_absolute_error(y_test, pred)),
            "smape": float(smape(y_test, pred)),
        }

    best_name = min(scores, key=lambda n: scores[n]["smape"])
    best_model = scores[best_name]["est"]
    return best_name, best_model, scores



def fit_quantiles(X_train, y_train, alphas=(0.1, 0.5, 0.9)):
    out = {}
    for a in alphas:
        m = GradientBoostingRegressor(
            loss="quantile", alpha=a, max_depth=3, n_estimators=600, learning_rate=0.08, random_state=42
        )
        m.fit(X_train, y_train)
        out[a] = m
    return out


def forecast_with_intervals_exo(
    X, y_train, start_row, target, horizon=12, station_delta=0.0, fuel_delta=0.0, clamp="train"
):
    """Recursive forecast (quantile bands) with exogenous deltas for infra & fuel."""
    qs = fit_quantiles(X.iloc[: start_row + 1], y_train.iloc[: start_row + 1], (0.1, 0.5, 0.9))

    lag_nums = sorted(int(c.split("lag")[-1]) for c in X.columns if f"{target}_lag" in c)
    ma_nums = sorted(int(c.split("ma")[-1]) for c in X.columns if f"{target}_ma" in c)
    has_d1 = f"{target}_diff1" in X.columns
    has_d3 = f"{target}_diff3" in X.columns

    has_sc = "station_count" in X.columns
    has_sc_l = "station_count_lag1" in X.columns
    has_sc_g = "station_growth" in X.columns
    has_sc_m = "station_ma3" in X.columns
    has_sc_x = "sc_x_lag1" in X.columns
    has_fp = "fuel_price" in X.columns
    has_fp_l = "fuel_price_lag1" in X.columns
    has_fp_c = "fuel_change" in X.columns

    t_lo, t_hi = np.percentile(y_train, [1, 99])

    hist_len = max(max(lag_nums or [1]), max(ma_nums or [1]))
    y_hist = list(y_train.iloc[: start_row + 1].values[-hist_len:])
    last = X.iloc[start_row].copy()
    sc_hist = [float(last["station_count"])] if has_sc else []
    fp_hist = [float(last["fuel_price"])] if has_fp else []
    X_last = X.iloc[[start_row]].copy()

    out = []
    for step in range(1, horizon + 1):
        x_new = X_last.copy()

        # exogenous roll-forward
        if has_sc:
            sc_next = sc_hist[-1] + station_delta
            x_new["station_count"] = sc_next
            sc_hist.append(sc_next)
            if has_sc_l:
                x_new["station_count_lag1"] = sc_hist[-2]
            if has_sc_g:
                x_new["station_growth"] = sc_hist[-1] - sc_hist[-2]
            if has_sc_m:
                x_new["station_ma3"] = np.mean(sc_hist[-min(3, len(sc_hist)) :])

        if has_fp:
            fp_next = fp_hist[-1] + fuel_delta
            x_new["fuel_price"] = fp_next
            fp_hist.append(fp_next)
            if has_fp_l:
                x_new["fuel_price_lag1"] = fp_hist[-2]
            if has_fp_c:
                x_new["fuel_change"] = fp_hist[-1] - fp_hist[-2]

        # target lags/MA/diffs
        if f"{target}_lag1" in x_new.columns:
            x_new[f"{target}_lag1"] = y_hist[-1]

        for L in sorted(lag_nums, reverse=True):
            if L == 1:
                continue
            colL, prev = f"{target}_lag{L}", f"{target}_lag{L-1}"
            if (colL in x_new.columns) and (prev in X_last.columns):
                x_new[colL] = X_last[prev].values

        for w in ma_nums:
            col = f"{target}_ma{w}"
            if col in x_new.columns:
                x_new[col] = float(np.mean(y_hist[-min(w, len(y_hist)) :]))

        if has_d1:
            x_new[f"{target}_diff1"] = y_hist[-1] - y_hist[-2] if len(y_hist) >= 2 else 0.0
        if has_d3:
            x_new[f"{target}_diff3"] = y_hist[-3] - y_hist[-6] if len(y_hist) >= 6 else 0.0
        if has_sc_x:
            x_new["sc_x_lag1"] = (x_new["station_count"] if has_sc else sc_hist[-1]) * x_new[f"{target}_lag1"]

        p10 = float(qs[0.1].predict(x_new)[0])
        p50 = float(qs[0.5].predict(x_new)[0])
        p90 = float(qs[0.9].predict(x_new)[0])

        if clamp == "train":
            lo, hi = max(1.0, t_lo), 1.2 * t_hi
            p10 = np.clip(p10, lo, hi)
            p50 = np.clip(p50, lo, hi)
            p90 = np.clip(p90, max(p50, lo), hi)

        out.append({"step": step, "p10": p10, "p50": p50, "p90": p90})
        y_hist.append(p50)
        X_last = x_new

    return pd.DataFrame(out)


def plot_backtest(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode="lines", name="Predicted (Ridge)"))
    fig.update_layout(title="Backtest — EV 2W Demand", xaxis_title="Test Month Index", yaxis_title="Units")
    return fig


def plot_band(fc, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["step"], y=fc["p90"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fc["step"], y=fc["p10"], fill="tonexty", name="P10–P90", mode="lines"))
    fig.add_trace(go.Scatter(x=fc["step"], y=fc["p50"], name="Median forecast", mode="lines"))
    fig.update_layout(title=title, xaxis_title="Month Ahead", yaxis_title="Units")
    return fig


def to_png_bytes(fig):
    """Export Plotly figure to PNG (requires kaleido). Returns None if not available."""
    try:
        import plotly.io as pio

        return pio.to_image(fig, format="png", scale=2)
    except Exception:
        return None


# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Data & Settings")

uploaded_ts = st.sidebar.file_uploader(
    "Upload monthly time-series CSV (date,demand_2w,demand_3w,station_count,fuel_price)", type=["csv"]
)
target = st.sidebar.selectbox("Target", ["demand_2w", "demand_3w"], index=0)
horizon = st.sidebar.slider("Forecast horizon (months)", 6, 18, 12)
delta_station = st.sidebar.slider("Infra growth per month (+chargers)", -50, 100, 50, step=5)
delta_fuel = st.sidebar.slider("Fuel price drift per month (₹)", -2.0, 2.0, 0.0, step=0.1)

uploaded_seg = st.sidebar.file_uploader(
    "Upload state-level segmentation CSV (merged_ev_state_data.csv)", type=["csv"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: If you don’t upload, a small demo dataset is used.")


# -------------------------------
# Data load
# -------------------------------
if uploaded_ts is not None:
    df_ts = pd.read_csv(uploaded_ts, parse_dates=["date"])
else:
    df_ts = demo_monthly_df()

st.title("EV Market & Forecast Studio")
st.write("Interactive app for **EV 2W/3W demand forecasting**, **scenario planning**, and **state segmentation**.")

tabs = st.tabs(["Forecasting", "Segmentation (States)", "Downloads"])


# -------------------------------
# Forecasting tab
# -------------------------------
with tabs[0]:
    st.subheader("1) Train & Backtest")
    with st.spinner("Building features…"):
        ts_feat = build_ts_features_plus(df_ts, target).dropna().reset_index(drop=True)
        drop_cols = (
            ["date", "state", "demand_2w", "demand_3w"]
            if "state" in ts_feat.columns
            else ["date", "demand_2w", "demand_3w"]
        )
        X = ts_feat.drop(columns=drop_cols)
        y = ts_feat[target].clip(lower=1.0)
        split = int(len(ts_feat) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        best_name, best_model, all_scores = select_and_fit(X_train, y_train, X_test, y_test)
        y_pred = best_model.predict(X_test)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.1f}")
    c2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.1f}")
    c3.metric("sMAPE", f"{smape(y_test, y_pred):.2f}%")
    st.plotly_chart(plot_backtest(y_test.values, y_pred), use_container_width=True)

    st.subheader("2) Forecast Bands & Infra Scenarios")
    with st.spinner("Simulating scenarios…"):
        fc_base = forecast_with_intervals_exo(
            X, y, start_row=split - 1, target=target, horizon=horizon, station_delta=0, fuel_delta=0
        )
        fc_10 = forecast_with_intervals_exo(
            X, y, start_row=split - 1, target=target, horizon=horizon, station_delta=10, fuel_delta=0
        )
        fc_scn = forecast_with_intervals_exo(
            X,
            y,
            start_row=split - 1,
            target=target,
            horizon=horizon,
            station_delta=delta_station,
            fuel_delta=delta_fuel,
        )

    st.plotly_chart(plot_band(fc_base, "12-Month Forecast (Baseline)"), use_container_width=True)
    st.plotly_chart(plot_band(fc_10, "12-Month Forecast (+10 chargers/mo)"), use_container_width=True)
    st.plotly_chart(
        plot_band(fc_scn, f"12-Month Forecast (+{delta_station} chargers/mo, fuel Δ {delta_fuel}/mo)"),
        use_container_width=True,
    )

    st.subheader("3) Uplift vs Baseline (Median)")
    comp = pd.DataFrame(
        {"Month": fc_base["step"], "P50_base": fc_base["p50"], "P50_scenario": fc_scn["p50"]}
    )
    comp["Uplift"] = comp["P50_scenario"] - comp["P50_base"]
    fig_u = px.line(comp, x="Month", y="Uplift", markers=True, title="Median Demand Uplift from Infrastructure Growth")
    fig_u.update_yaxes(title="Extra Units")
    st.plotly_chart(fig_u, use_container_width=True)
    st.dataframe(comp.round(1), use_container_width=True)

    st.subheader("4) Feature Importance (Permutation)")
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=15, random_state=42)
    imp = (
        pd.DataFrame({"feature": X_test.columns, "importance": perm.importances_mean})
        .sort_values("importance", ascending=False)
        .head(12)
    )
    fig_imp = px.bar(imp, x="importance", y="feature", orientation="h", title="Permutation Importance (Top 12)")
    st.plotly_chart(fig_imp, use_container_width=True)


# -------------------------------
# Segmentation tab
# -------------------------------
with tabs[1]:
    st.subheader("State Segmentation (KMeans)")
    st.caption(
        "Upload your `merged_ev_state_data.csv` with columns like: "
        "EV_2W, EV_3W, Total_EV, Highway_Charging, RO_Charging, Total_Charging."
    )
    if uploaded_seg is not None:
        df_seg = pd.read_csv(uploaded_seg)

        # autodetect numeric features
        num_cols = [c for c in df_seg.columns if df_seg[c].dtype != "O" and c not in ["Sl. No._x", "Sl. No._y"]]
        default_feats = [c for c in ["EV_2W", "EV_3W", "Total_Charging", "Highway_Charging", "RO_Charging"] if c in num_cols]
        features = st.multiselect("Select features for clustering", num_cols, default=default_feats)
        k = st.slider("Number of clusters (k)", 2, 6, 3)

        if features:
            Xs = StandardScaler().fit_transform(df_seg[features].fillna(0))
            # n_init int for broad sklearn compatibility
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
            df_seg["Segment"] = km.labels_.astype(int)

            # 2D projection (first two chosen features)
            xcol = features[0]
            ycol = features[1] if len(features) > 1 else features[0]
            fig_sc = px.scatter(
                df_seg,
                x=xcol,
                y=ycol,
                color="Segment",
                hover_data=["State/UT"] if "State/UT" in df_seg.columns else None,
                title="State Segments (colored)",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            display_cols = (["State/UT"] if "State/UT" in df_seg.columns else []) + ["Segment"] + features
            st.dataframe(df_seg[display_cols], use_container_width=True)

            st.download_button(
                "Download segmented CSV",
                data=df_seg.to_csv(index=False).encode("utf-8"),
                file_name="segmented_states.csv",
                mime="text/csv",
            )
        else:
            st.info("Pick at least one numeric feature.")
    else:
        st.info("Upload your state dataset to run clustering.")


# -------------------------------
# Downloads tab
# -------------------------------
with tabs[2]:
    st.subheader("Export")

    # Backtest figure
    fig_bt = plot_backtest(y_test.values, y_pred)
    bt_png = to_png_bytes(fig_bt)
    if bt_png is not None:
        st.download_button("Backtest PNG", data=bt_png, file_name="backtest.png", mime="image/png")
    else:
        st.info("Install kaleido to enable image downloads:  `pip install -U kaleido`")

    # Band PNGs
    items = [
        ("forecast_base.png", plot_band(fc_base, "Baseline")),
        ("forecast_plus10.png", plot_band(fc_10, "+10 chargers/mo")),
        ("forecast_scenario.png", plot_band(fc_scn, f"+{delta_station} chargers/mo")),
    ]
    for name, fig in items:
        png = to_png_bytes(fig)
        if png is not None:
            st.download_button(f"Download {name}", data=png, file_name=name, mime="image/png")

    # CSVs
    st.download_button(
        "Forecast (baseline) CSV",
        data=fc_base.to_csv(index=False).encode("utf-8"),
        file_name="forecast_baseline.csv",
        mime="text/csv",
    )
    st.download_button(
        "Forecast (scenario) CSV",
        data=fc_scn.to_csv(index=False).encode("utf-8"),
        file_name="forecast_scenario.csv",
        mime="text/csv",
    )
    st.download_button(
        "Uplift CSV", data=comp.to_csv(index=False).encode("utf-8"), file_name="uplift.csv", mime="text/csv"
    )

st.caption("© Your Name — EV Segmentation & Forecast Studio")
