# Ev-market-forecast-studio
Interactive EV 2W/3W demand forecasting, scenario planning, and state segmentation (Streamlit + notebook).

Interactive studio for EV 2W/3W demand forecasting, scenario planning (e.g., charger rollouts), and state segmentation—built with Streamlit + scikit-learn.
It combines a research notebook and a production-style app so you can explore, forecast, and communicate results in one place.
# What this does

This project builds a practical EV market analytics studio focused on India’s EV two-wheelers (2W) and three-wheelers (3W).
It combines:

Leak-safe time-series forecasting (with quantile prediction bands),

What-if scenarios for charging-infrastructure growth,

K-Means state segmentation for market tiering.

The studio comes in two forms:

Streamlit app (app1.py) for interactive use.

Jupyter notebook (analysis + reproducible plots).
# Technologies

Project is built with:

Python 3.9+

pandas, numpy

scikit-learn

matplotlib, seaborn, plotly

streamlit
# Technologies

Project is built with:

Python 3.9+

pandas, numpy

scikit-learn

matplotlib, seaborn, plotly

streamlit
# Usage
#Time-series forecasting

Go to the Forecasting tab.

Upload (or use the demo) time-series CSV with columns:

date (month start), demand_2w, demand_3w, station_count, fuel_price.

Choose Target (demand_2w or demand_3w).

The app:

Builds leak-safe lag/MA/diff features,

Trains multiple models with TimeSeriesSplit,

Selects the best by sMAPE,

Shows a backtest plot,

Produces 12-month quantile forecasts (P10–P50–P90).

# Scenario planning

In the Forecast Bands & Infra Scenarios section, adjust:

Infra growth per month (+chargers) (e.g., +10 or +50),

Fuel price drift (optional).

The app simulates future station_count trajectories and recomputes forecasts, then plots:

Baseline vs scenario forecast bands,

Median uplift vs baseline by month,

A small table with P50 and uplift values.

# State segmentation

Switch to Segmentation (States).

Upload a CSV like merged_ev_state_data.csv that includes feature columns:

e.g., EV_2W, EV_3W, Total_EV, Highway_Charging, RO_Charging, Total_Charging.

Pick numeric features, choose k (2–6), and run K-Means.

You’ll get:

A colored 2D plot of segments,

A segment profile table (mean values),

Option to download the segmented CSV.
