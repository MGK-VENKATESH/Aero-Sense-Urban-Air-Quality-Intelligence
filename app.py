
import os
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from textwrap import dedent

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AeroSense | Urban Air Quality Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. DASHBOARD THEME
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --background: #0B1120;
    --sidebar: #101A2A;
    --card: #172438;
    --card-light: #1B2B40;
    --border: #26364B;
    --text: #F1F5F9;
    --muted: #94A3B8;
    --accent: #2DD4BF;
    --accent-dark: #14B8A6;
    --red: #FF555D;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    font-weight: 700 !important;
}

p, label, span {
    color: inherit;
}

[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 800;
}

.stSelectbox > div > div,
.stTextInput > div > div {
    background: #151D2B;
    border-color: var(--border);
    border-radius: 8px;
}

.stSlider [data-baseweb="slider"] {
    padding-top: 5px;
}

.stButton > button {
    background: var(--accent-dark);
    color: #06151A;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 10px 15px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: var(--accent);
    color: #06151A;
    border: none;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stPlotlyChart"],
[data-testid="stImage"] {
    border-radius: 12px;
}

hr {
    border-color: var(--border);
    margin-top: 2rem;
    margin-bottom: 2rem;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

.aero-logo {
    font-size: 25px;
    font-weight: 800;
    color: #2DD4BF;
    letter-spacing: -1px;
}

.aero-tagline {
    font-size: 9px;
    color: #94A3B8;
    letter-spacing: 2px;
    margin-top: 4px;
}

.section-label {
    color: #2DD4BF;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 9px;
}

.forecast-card {
    background: linear-gradient(145deg, #172438, #111D30);
    border: 1px solid #293B52;
    border-radius: 13px;
    padding: 22px 20px;
    min-height: 160px;
    height: 100%;
}

.forecast-card:hover {
    border-color: #2DD4BF;
    transition: 0.2s;
}

.forecast-label {
    font-size: 11px;
    color: #94A3B8;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 18px;
}

.forecast-value {
    font-size: 35px;
    font-weight: 800;
    color: #F1F5F9;
    margin-bottom: 9px;
}

.forecast-category {
    font-size: 12px;
    font-weight: 600;
}

.info-card {
    background: #172438;
    border: 1px solid #293B52;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 13px;
}

.info-title {
    color: #94A3B8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.info-value {
    color: #F1F5F9;
    font-size: 26px;
    font-weight: 800;
}

.ranking-card {
    background: #172438;
    border: 1px solid #293B52;
    border-radius: 11px;
    padding: 16px;
    margin-bottom: 12px;
}

.ranking-number {
    color: #2DD4BF;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.ranking-name {
    color: #F1F5F9;
    font-size: 15px;
    font-weight: 700;
    margin-top: 7px;
}

.ranking-score {
    color: #94A3B8;
    font-size: 11px;
    margin-top: 6px;
}

.feature-card {
    background: #172438;
    border: 1px solid #293B52;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
}

.feature-name {
    color: #F1F5F9;
    font-size: 13px;
    font-weight: 700;
}

.feature-score {
    color: #2DD4BF;
    font-size: 12px;
    margin-top: 5px;
}

.notice-card {
    background: #192E49;
    border: 1px solid #294968;
    color: #B8C9DF;
    border-radius: 9px;
    padding: 14px;
    font-size: 12px;
    line-height: 1.7;
}

.footer {
    text-align: center;
    color: #64748B;
    font-size: 11px;
    padding: 25px 0 5px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def render_html(content):
    """Render complete HTML blocks safely."""
    st.html(dedent(content).strip())


def get_aqi_category(aqi):
    """Return AQI category, icon and display color."""

    if aqi <= 50:
        return "Good", "🟢", "#22C55E"

    elif aqi <= 100:
        return "Satisfactory", "🟡", "#EAB308"

    elif aqi <= 200:
        return "Moderate", "🟠", "#F97316"

    elif aqi <= 300:
        return "Poor", "🔴", "#EF4444"

    elif aqi <= 400:
        return "Very Poor", "🟣", "#A855F7"

    else:
        return "Severe", "🟤", "#A16207"


def safe_float(value, default=0.0):
    """Convert values safely into float."""
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (ValueError, TypeError):
        return default


# ============================================================
# 4. DATA LOADING AND FEATURE ENGINEERING
# ============================================================

REQUIRED_COLUMNS = [
    "City",
    "Date",
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "AQI"
]

FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "AQI",
    "Month",
    "DayOfWeek",
    "AQI_lag1",
    "AQI_lag2",
    "AQI_lag7",
    "AQI_roll3",
    "AQI_roll7"
]

TARGETS = [
    "Target_1D",
    "Target_3D",
    "Target_7D"
]

HORIZONS = [
    "+1 Day",
    "+3 Days",
    "+7 Days"
]


@st.cache_data(show_spinner=False)
def load_raw_data(csv_path, modified_time):
    """Load the original dataset."""

    df = pd.read_csv(csv_path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    numeric_columns = [
        "PM2.5", "PM10", "NO2", "CO",
        "SO2", "O3", "AQI"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(subset=["City", "Date"])

    return df


@st.cache_data(show_spinner=False)
def load_and_prepare_data(csv_path, modified_time, city):

    raw_df = load_raw_data(csv_path, modified_time)

    df = raw_df[raw_df["City"] == city].copy()

    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(
            "Not enough records available for this city."
        )

    # Keep only required columns
    df = df[["Date"] + FEATURES[:7]].copy()

    # Forward-fill missing observations.
    # Do not backfill using future observations.
    df[FEATURES[:7]] = df[FEATURES[:7]].ffill()

    df = df.dropna().reset_index(drop=True)

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    df["Month"] = df["Date"].dt.month

    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df["AQI_lag1"] = df["AQI"].shift(1)

    df["AQI_lag2"] = df["AQI"].shift(2)

    df["AQI_lag7"] = df["AQI"].shift(7)

    df["AQI_roll3"] = df["AQI"].rolling(
        window=3
    ).mean()

    df["AQI_roll7"] = df["AQI"].rolling(
        window=7
    ).mean()

    # --------------------------------------------------------
    # Multi-Horizon Forecast Targets
    # --------------------------------------------------------

    df["Target_1D"] = df["AQI"].shift(-1)

    df["Target_3D"] = df["AQI"].shift(-3)

    df["Target_7D"] = df["AQI"].shift(-7)

    # Actual target dates for plotting
    df["TargetDate_1D"] = df["Date"].shift(-1)

    df["TargetDate_3D"] = df["Date"].shift(-3)

    df["TargetDate_7D"] = df["Date"].shift(-7)

    df = df.dropna().reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(
            "Insufficient usable records after feature engineering."
        )

    return df


# ============================================================
# 5. MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def train_models(city_df, city):

    X = city_df[FEATURES].copy()

    Y = city_df[TARGETS].copy()

    split = int(len(city_df) * 0.8)

    if split < 10 or len(city_df) - split < 5:
        raise ValueError(
            "Not enough training or testing records."
        )

    X_train = X.iloc[:split]

    X_test = X.iloc[split:]

    Y_train = Y.iloc[:split]

    Y_test = Y.iloc[split:]

    test_dates = {
        "+1 Day": city_df["TargetDate_1D"].iloc[split:],
        "+3 Days": city_df["TargetDate_3D"].iloc[split:],
        "+7 Days": city_df["TargetDate_7D"].iloc[split:]
    }

    models = {

        "Random Forest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        ),

        "Decision Tree": MultiOutputRegressor(
            DecisionTreeRegressor(
                max_depth=6,
                min_samples_split=5,
                random_state=42
            )
        ),

        "KNN Regressor": make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(
                KNeighborsRegressor(
                    n_neighbors=7,
                    weights="distance"
                )
            )
        ),

        "Ridge Regression": make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(
                Ridge(alpha=1.0)
            )
        )
    }

    results = {}

    metrics_rows = []

    # --------------------------------------------------------
    # Train and Evaluate Each Model
    # --------------------------------------------------------

    for name, model in models.items():

        model.fit(X_train, Y_train)

        predictions = model.predict(X_test)

        results[name] = {
            "model": model,
            "predictions": predictions
        }

        for i, horizon in enumerate(HORIZONS):

            actual = Y_test.iloc[:, i].values

            predicted = predictions[:, i]

            mae = mean_absolute_error(
                actual,
                predicted
            )

            rmse = np.sqrt(
                mean_squared_error(
                    actual,
                    predicted
                )
            )

            r2 = r2_score(
                actual,
                predicted
            )

            metrics_rows.append({

                "Model": name,

                "Horizon": horizon,

                "MAE": round(mae, 2),

                "RMSE": round(rmse, 2),

                "R² Score": round(r2, 3)

            })

    metrics_df = pd.DataFrame(metrics_rows)

    return (
        results,
        metrics_df,
        X,
        Y_test,
        X_test,
        test_dates,
        split
    )


# ============================================================
# 6. SIDEBAR - BRANDING AND CONFIGURATION
# ============================================================

with st.sidebar:

    render_html("""
    <div style="padding:5px 0 25px 0;">
        <div class="aero-logo">◈ AEROSENSE</div>
        <div class="aero-tagline">
            AIR QUALITY INTELLIGENCE
        </div>
    </div>
    """)

    st.divider()

    st.subheader("⚙️ Configuration")

    csv_path = st.text_input(
        "Dataset CSV Path",
        value="city_day.csv",
        help="Enter the complete path to your city_day.csv file."
    )

    st.caption(
        "Example: /Users/yourname/Documents/city_day.csv"
    )


# ============================================================
# 7. DATASET VALIDATION
# ============================================================

if not os.path.isfile(csv_path):

    st.error(
        "Dataset not found. Please enter a valid path to city_day.csv."
    )

    st.info(
        "You can also place city_day.csv in the same folder as app.py "
        "and use the path city_day.csv."
    )

    st.stop()


try:

    modified_time = os.path.getmtime(csv_path)

    raw_df = load_raw_data(
        csv_path,
        modified_time
    )

    valid_cities = (

        raw_df.dropna(
            subset=["AQI", "PM2.5"]
        )

        .groupby("City")

        .size()

        .loc[lambda x: x > 300]

        .index.tolist()

    )

    valid_cities = sorted(valid_cities)

    if not valid_cities:

        st.error(
            "No cities have more than 300 valid AQI and PM2.5 records."
        )

        st.stop()

except Exception as error:

    st.error(f"Dataset loading error: {error}")

    st.stop()


# ============================================================
# 8. SIDEBAR - MODEL AND CITY SELECTION
# ============================================================

with st.sidebar:

    selected_city = st.selectbox(
        "Select City",
        valid_cities
    )

    model_names = [
        "Random Forest",
        "Decision Tree",
        "KNN Regressor",
        "Ridge Regression"
    ]

    selected_model_name = st.selectbox(
        "Select ML Algorithm",
        model_names
    )

    st.divider()

    st.subheader("🧪 What-If Simulator")

    st.caption(
        "Change pollutant values to simulate hypothetical AQI scenarios."
    )


# ============================================================
# 9. PREPARE CITY DATA AND TRAIN MODELS
# ============================================================

try:

    with st.spinner(
        f"Preparing data and training models for {selected_city}..."
    ):

        df = load_and_prepare_data(
            csv_path,
            modified_time,
            selected_city
        )

        (
            results,
            metrics_df,
            X,
            Y_test,
            X_test,
            test_dates,
            split
        ) = train_models(
            df,
            selected_city
        )

except Exception as error:

    st.error(f"Model training error: {error}")

    st.stop()


active_model = results[selected_model_name]["model"]


# ============================================================
# 10. WHAT-IF SIMULATOR
# ============================================================

latest_row = X.iloc[-1].copy()

latest_date = df["Date"].iloc[-1]

default_values = {

    "sim_pm25": safe_float(latest_row["PM2.5"]),

    "sim_pm10": safe_float(latest_row["PM10"]),

    "sim_no2": safe_float(latest_row["NO2"]),

    "sim_co": safe_float(latest_row["CO"]),

    "sim_aqi": safe_float(latest_row["AQI"])

}

slider_config = [

    ("sim_pm25", "PM2.5 (µg/m³)", 0.0, 500.0, 1),

    ("sim_pm10", "PM10 (µg/m³)", 0.0, 600.0, 1),

    ("sim_no2", "NO2 (µg/m³)", 0.0, 200.0, 1),

    ("sim_co", "CO (mg/m³)", 0.0, 10.0, 2),

    ("sim_aqi", "Current AQI", 0.0, 500.0, 1)

]

with st.sidebar:

    # Initialize simulator values
    for key, _, minimum, maximum, _ in slider_config:

        if key not in st.session_state:

            st.session_state[key] = float(
                np.clip(
                    default_values[key],
                    minimum,
                    maximum
                )
            )

    # Reset button must be above the sliders.
    if st.button(
        "↺ Reset Simulator",
        use_container_width=True
    ):

        for key, _, minimum, maximum, _ in slider_config:

            st.session_state[key] = float(
                np.clip(
                    default_values[key],
                    minimum,
                    maximum
                )
            )

    simulator_values = {}

    for key, label, minimum, maximum, decimal in slider_config:

        simulator_values[key] = st.slider(
            label,
            min_value=minimum,
            max_value=maximum,
            value=float(st.session_state[key]),
            step=0.1 if decimal == 1 else 0.01,
            key=key
        )


# Build What-If Input
input_sample = latest_row.copy()

input_sample["PM2.5"] = simulator_values["sim_pm25"]

input_sample["PM10"] = simulator_values["sim_pm10"]

input_sample["NO2"] = simulator_values["sim_no2"]

input_sample["CO"] = simulator_values["sim_co"]

input_sample["AQI"] = simulator_values["sim_aqi"]

forecast = active_model.predict(
    pd.DataFrame(
        [input_sample],
        columns=FEATURES
    )
)[0]

curr_aqi = simulator_values["sim_aqi"]


# ============================================================
# 11. DASHBOARD HEADER
# ============================================================

render_html(f"""
<div style="
    background:linear-gradient(115deg,#172B3D,#153A3A);
    border:1px solid #244B53;
    border-radius:14px;
    padding:30px;
    margin-bottom:28px;
">

    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        flex-wrap:wrap;
        gap:12px;
    ">

        <div style="
            color:#2DD4BF;
            font-size:13px;
            font-weight:800;
            letter-spacing:2px;
        ">
            ◈ AEROSENSE / ANALYTICS
        </div>

        <div style="
            color:#2DD4BF;
            font-size:11px;
            font-weight:700;
        ">
            ● SYSTEM ONLINE
        </div>

    </div>

    <div style="
        font-size:32px;
        font-weight:800;
        color:#F1F5F9;
        margin-top:30px;
        letter-spacing:-1px;
    ">
        Urban Air Quality Intelligence
    </div>

    <div style="
        color:#B8C9DF;
        font-size:13px;
        margin-top:12px;
        line-height:1.8;
    ">
        Multi-horizon AQI forecasting |
        Machine learning based air quality monitoring
    </div>

    <div style="
        margin-top:25px;
        color:#94A3B8;
        font-size:12px;
    ">
        CITY:
        <b style="color:#2DD4BF;">{selected_city}</b>
        &nbsp; | &nbsp;
        MODEL:
        <b style="color:#2DD4BF;">{selected_model_name}</b>
        &nbsp; | &nbsp;
        LATEST DATA:
        <b style="color:#2DD4BF;">{latest_date.strftime('%d %b %Y')}</b>
    </div>

</div>
""")


# ============================================================
# 12. SECTION 01 - AQI FORECAST OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-label">01 / AQI OVERVIEW</div>',
    unsafe_allow_html=True
)

st.header("Air Quality Forecast Summary")

curr_cat, curr_icon, curr_color = get_aqi_category(curr_aqi)

col1, col2, col3, col4 = st.columns(4)

# Current AQI Card
with col1:

    render_html(f"""
    <div class="forecast-card">

        <div class="forecast-label">
            CURRENT AQI
        </div>

        <div class="forecast-value">
            {curr_aqi:.0f}
        </div>

        <div class="forecast-category"
             style="color:{curr_color};">
            {curr_icon} {curr_cat}
        </div>

    </div>
    """)


# Forecast Cards
forecast_columns = [col2, col3, col4]

for i, (column, horizon, value) in enumerate(
    zip(forecast_columns, HORIZONS, forecast)
):

    category, icon, color = get_aqi_category(value)

    delta = value - curr_aqi

    with column:

        render_html(f"""
        <div class="forecast-card">

            <div class="forecast-label">
                {horizon.upper()} FORECAST
            </div>

            <div class="forecast-value">
                {value:.0f}
            </div>

            <div class="forecast-category"
                 style="color:{color};">
                {icon} {category}
            </div>

            <div style="
                color:#94A3B8;
                font-size:11px;
                margin-top:12px;
            ">
                {delta:+.0f} AQI from current input
            </div>

        </div>
        """)


st.markdown("")

render_html("""
<div class="notice-card">
    <b>Forecast Disclaimer:</b>
    These are model-generated What-If estimates using the selected
    pollutant inputs and latest available historical feature values.
    They are not live sensor readings or guaranteed future conditions.
</div>
""")


# ============================================================
# 13. SECTION 02 - ACTUAL VS PREDICTED AQI
# ============================================================

st.divider()

st.markdown(
    '<div class="section-label">02 / FORECAST ANALYSIS</div>',
    unsafe_allow_html=True
)

st.header("Actual vs Predicted AQI")

left_col, right_col = st.columns([1.8, 1])

with left_col:

    horizon_choice = st.selectbox(
        "Select Prediction Horizon",
        HORIZONS
    )

    horizon_index = HORIZONS.index(horizon_choice)

    predictions = results[
        selected_model_name
    ]["predictions"][:, horizon_index]

    actual = Y_test.iloc[:, horizon_index].values

    target_dates = test_dates[horizon_choice]

    plot_count = min(120, len(actual))

    fig, ax = plt.subplots(figsize=(11, 4.5))

    fig.patch.set_facecolor("#111D30")

    ax.set_facecolor("#111D30")

    ax.plot(
        target_dates.iloc[-plot_count:],
        actual[-plot_count:],
        color="#2DD4BF",
        label="Actual AQI",
        linewidth=2.2
    )

    ax.plot(
        target_dates.iloc[-plot_count:],
        predictions[-plot_count:],
        color="#F59E0B",
        label="Predicted AQI",
        linestyle="--",
        linewidth=2
    )

    ax.set_title(
        f"{horizon_choice} Forecast Performance",
        color="#F1F5F9",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=15
    )

    ax.set_xlabel(
        "Target Date",
        color="#CBD5E1"
    )

    ax.set_ylabel(
        "AQI",
        color="#CBD5E1"
    )

    ax.tick_params(
        colors="#CBD5E1",
        labelsize=9
    )

    ax.grid(
        alpha=0.15,
        color="#94A3B8"
    )

    for spine in ax.spines.values():

        spine.set_color("#293B52")

    ax.legend(
        facecolor="#172438",
        edgecolor="#293B52",
        labelcolor="#F1F5F9"
    )

    fig.autofmt_xdate()

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


with right_col:

    st.subheader("Prediction Statistics")

    latest_actual = float(actual[-1])

    latest_predicted = float(predictions[-1])

    latest_difference = latest_predicted - latest_actual

    latest_actual_cat, latest_actual_icon, _ = get_aqi_category(
        latest_actual
    )

    st.metric(
        "Latest Actual AQI",
        f"{latest_actual:.2f}",
        latest_actual_cat
    )

    st.metric(
        "Latest Predicted AQI",
        f"{latest_predicted:.2f}",
        f"{latest_difference:+.2f}",
        delta_color="inverse"
    )

    selected_metrics = metrics_df[
        (metrics_df["Model"] == selected_model_name)
        & (metrics_df["Horizon"] == horizon_choice)
    ].iloc[0]

    render_html(f"""
    <div class="info-card">

        <div class="info-title">
            MEAN ABSOLUTE ERROR
        </div>

        <div class="info-value">
            {selected_metrics["MAE"]:.2f}
        </div>

        <hr>

        <div class="info-title">
            ROOT MEAN SQUARED ERROR
        </div>

        <div class="info-value">
            {selected_metrics["RMSE"]:.2f}
        </div>

        <hr>

        <div class="info-title">
            R² SCORE
        </div>

        <div class="info-value">
            {selected_metrics["R² Score"]:.3f}
        </div>

    </div>
    """)


# ============================================================
# 14. SECTION 03 - MODEL PERFORMANCE BENCHMARK
# ============================================================

st.divider()

st.markdown(
    '<div class="section-label">03 / MODEL PERFORMANCE</div>',
    unsafe_allow_html=True
)

st.header("Machine Learning Model Benchmark")

st.caption(
    "Performance comparison across all algorithms and forecast horizons."
)

benchmark_col, ranking_col = st.columns([1.8, 1])

with benchmark_col:

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
        height=390,
        column_config={

            "MAE": st.column_config.NumberColumn(
                "MAE",
                format="%.2f"
            ),

            "RMSE": st.column_config.NumberColumn(
                "RMSE",
                format="%.2f"
            ),

            "R² Score": st.column_config.NumberColumn(
                "R² Score",
                format="%.3f"
            )
        }
    )


# ============================================================
# 15. ALGORITHM RANKING
# ============================================================

with ranking_col:

    st.subheader("Algorithm Ranking")

    ranking_df = (

        metrics_df

        .groupby("Model")

        .agg(
            Average_MAE=("MAE", "mean"),
            Average_RMSE=("RMSE", "mean")
        )

        .sort_values("Average_MAE")

        .reset_index()

    )

    for rank, row in ranking_df.iterrows():

        if rank == 0:
            rank_label = "RANK #1"
            rank_color = "#2DD4BF"

        elif rank == 1:
            rank_label = "RANK #2"
            rank_color = "#CBD5E1"

        else:
            rank_label = f"RANK #{rank + 1}"
            rank_color = "#94A3B8"

        render_html(f"""
        <div class="ranking-card">

            <div class="ranking-number"
                 style="color:{rank_color};">
                {rank_label}
            </div>

            <div class="ranking-name">
                {row["Model"]}
            </div>

            <div class="ranking-score">
                Average MAE:
                <b style="color:#2DD4BF;">
                    {row["Average_MAE"]:.2f}
                </b>
            </div>

        </div>
        """)

    st.caption(
        "Ranking is based on the average MAE across all three horizons. "
        "Lower MAE indicates better performance."
    )


# ============================================================
# 16. MODEL COMPARISON CHART
# ============================================================

st.subheader("Model Performance Comparison")

comparison_metric = st.selectbox(
    "Select Evaluation Metric",
    ["MAE", "RMSE", "R² Score"]
)

comparison_pivot = metrics_df.pivot(
    index="Model",
    columns="Horizon",
    values=comparison_metric
)

fig_compare, ax_compare = plt.subplots(figsize=(11, 4.5))

fig_compare.patch.set_facecolor("#111D30")

ax_compare.set_facecolor("#111D30")

comparison_pivot.plot(
    kind="bar",
    ax=ax_compare,
    width=0.75
)

ax_compare.set_title(
    f"Model Comparison - {comparison_metric}",
    color="#F1F5F9",
    fontsize=13,
    fontweight="bold"
)

ax_compare.set_xlabel(
    "Machine Learning Algorithm",
    color="#CBD5E1"
)

ax_compare.set_ylabel(
    comparison_metric,
    color="#CBD5E1"
)

ax_compare.tick_params(
    axis="x",
    rotation=0,
    colors="#CBD5E1"
)

ax_compare.tick_params(
    axis="y",
    colors="#CBD5E1"
)

ax_compare.grid(
    axis="y",
    alpha=0.15
)

ax_compare.legend(
    facecolor="#172438",
    edgecolor="#293B52",
    labelcolor="#F1F5F9"
)

for spine in ax_compare.spines.values():

    spine.set_color("#293B52")

plt.tight_layout()

st.pyplot(fig_compare, use_container_width=True)

plt.close(fig_compare)


# ============================================================
# 17. SECTION 04 - FEATURE IMPORTANCE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-label">04 / MODEL INTERPRETABILITY</div>',
    unsafe_allow_html=True
)

st.header("Feature Importance Analysis")

st.caption(
    "Identifying the most influential input features for AQI forecasting."
)

if selected_model_name in ["Random Forest", "Decision Tree"]:

    importance_col, insight_col = st.columns([1.8, 1])

    estimator = active_model.estimators_[0]

    importances = estimator.feature_importances_

    feature_df = pd.DataFrame({

        "Feature": FEATURES,

        "Importance": importances

    }).sort_values(
        "Importance",
        ascending=True
    )

    with importance_col:

        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))

        fig_imp.patch.set_facecolor("#111D30")

        ax_imp.set_facecolor("#111D30")

        ax_imp.barh(
            feature_df["Feature"],
            feature_df["Importance"],
            color="#2DD4BF",
            alpha=0.9
        )

        ax_imp.set_title(
            "Feature Contribution (+1 Day Horizon)",
            color="#F1F5F9",
            fontsize=13,
            fontweight="bold",
            loc="left",
            pad=15
        )

        ax_imp.set_xlabel(
            "Relative Importance",
            color="#CBD5E1"
        )

        ax_imp.set_ylabel(
            "Input Features",
            color="#CBD5E1"
        )

        ax_imp.tick_params(
            colors="#CBD5E1",
            labelsize=9
        )

        ax_imp.grid(
            axis="x",
            alpha=0.15
        )

        for spine in ax_imp.spines.values():

            spine.set_color("#293B52")

        plt.tight_layout()

        st.pyplot(
            fig_imp,
            use_container_width=True
        )

        plt.close(fig_imp)

    with insight_col:

        st.subheader("Top Influential Features")

        top_features = feature_df.sort_values(
            "Importance",
            ascending=False
        ).head(5)

        for _, row in top_features.iterrows():

            render_html(f"""
            <div class="feature-card">

                <div class="feature-name">
                    {row["Feature"]}
                </div>

                <div class="feature-score">
                    Importance: {row["Importance"]:.4f}
                </div>

            </div>
            """)

        st.caption(
            "Feature importance reflects the selected tree model's "
            "split-based importance, not causal influence."
        )

else:

    st.info(
        "Feature importance visualization is available for "
        "Random Forest and Decision Tree models. "
        "Ridge Regression and KNN do not provide this tree-based "
        "feature importance attribute."
    )


# ============================================================
# 18. DATASET INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-label">05 / DATASET INFORMATION</div>',
    unsafe_allow_html=True
)

st.header("Dataset Overview")

data_col1, data_col2, data_col3, data_col4 = st.columns(4)

with data_col1:

    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with data_col2:

    st.metric(
        "Training Records",
        f"{split:,}"
    )

with data_col3:

    st.metric(
        "Testing Records",
        f"{len(df) - split:,}"
    )

with data_col4:

    st.metric(
        "Input Features",
        len(FEATURES)
    )


st.subheader("Recent Historical Records")

display_columns = [
    "Date",
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "AQI"
]

st.dataframe(
    df[display_columns].tail(10),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 19. FOOTER
# ============================================================

render_html("""
<div class="footer">

    <div style="
        color:#2DD4BF;
        font-weight:800;
        font-size:13px;
        letter-spacing:2px;
        margin-bottom:8px;
    ">
        ◈ AEROSENSE
    </div>

    Urban Air Quality Intelligence and Multi-Horizon Forecasting

    <br><br>

    Powered by Machine Learning | Random Forest | Decision Tree |
    KNN | Ridge Regression

    <br><br>

    Research and Academic Demonstration Dashboard

</div>
""")
