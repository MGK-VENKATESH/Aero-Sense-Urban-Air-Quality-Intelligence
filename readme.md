# ◈ AeroSense: Urban Air Quality Intelligence
### Multi-Horizon Air Quality Index (AQI) Forecasting Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-MultiOutput-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Course](https://img.shields.io/badge/M.Tech%20AI%26DS-26CS5104E-2DD4BF)](#academic-information)

**AeroSense** is an interactive machine learning analytical dashboard designed to analyze historical urban air quality records and forecast Air Quality Index (AQI) values across multiple future time horizons (**+1 Day**, **+3 Days**, and **+7 Days**). 

Built with **Python**, **Scikit-Learn**, and **Streamlit**, the platform pairs multi-output regression modeling with an interactive **What-If Pollutant Simulator**, model performance benchmarks, and tree-based feature interpretability.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Machine Learning Models](#-machine-learning-models)
- [Dataset & Feature Engineering](#-dataset--feature-engineering)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Repository Structure](#-repository-structure)
- [Limitations & Disclaimer](#-limitations--disclaimer)
- [Future Enhancements](#-future-enhancements)
- [Team & Academic Information](#-team--academic-information)

---

## 🌟 Project Overview
Urban AQI fluctuates dynamically due to vehicular emissions, industrial activity, seasonal inversions, and local meteorological patterns. Single-step forecasting models only estimate immediate next-day conditions, leaving minimal lead time for planning. 

**AeroSense** addresses this by framing AQI forecasting as a **multi-horizon regression problem**, enabling simultaneous estimation of **+1-day**, **+3-day**, and **+7-day** AQI values alongside Indian National AQI health advisory categories (*Good, Satisfactory, Moderate, Poor, Very Poor, Severe*).

---

## 🚀 Key Features
* **Multi-Horizon AQI Forecasting:** Predicts AQI simultaneously for **+1 Day**, **+3 Days**, and **+7 Days** using `MultiOutputRegressor`.
* **Interactive What-If Pollutant Simulator:** Adjust real-time sliders for `PM2.5`, `PM10`, `NO2`, `CO`, and `Current AQI` to explore hypothetical pollution scenarios and observe model-generated forecast shifts.
* **4-Algorithm Comparative Benchmark:** Evaluates and ranks **Random Forest**, **Decision Tree**, **K-Nearest Neighbors (KNN)**, and **Ridge Regression** across **MAE**, **RMSE**, and **$R^2$ Score**.
* **Visual Forecast Diagnostics:** Interactive **Actual vs. Predicted AQI** time-series plots aligned to true target dates across all three forecast horizons.
* **Model Interpretability:** Split-based **Feature Importance Analysis** for tree-based models (`Random Forest` and `Decision Tree`) highlighting key drivers of near-term AQI.
* **City-Level Filtering:** Supports any city in `city_day.csv` with more than 300 valid non-missing `AQI` and `PM2.5` observations.

---

## 🏗️ System Architecture & Pipeline

```text
city_day.csv ──▶ City Filtering & Chronological Sorting ──▶ Forward-Fill Imputation
                                                                      │
Streamlit Dashboard ◀── Evaluation (MAE, RMSE, R²) ◀── MultiOutputRegressor (4 Models) ◀── Feature Engineering (14 Features)
```

1. **Data Preparation:** Filters records by selected city, parses timestamps chronologically, and applies forward-fill (`ffill`) imputation without future data leakage.
2. **Feature Engineering:** Generates 14 predictive features combining raw pollutants, calendar indicators, historical AQI lags, and moving averages.
3. **Chronological Train/Test Split:** Splits city records into **80% training** and **20% testing** sets in strict time order.
4. **Multi-Output Training:** Wraps each base regressor in `MultiOutputRegressor` to fit dedicated estimators for `+1 Day`, `+3 Days`, and `+7 Days` targets.

---

## 🤖 Machine Learning Models

| Algorithm | Configuration | Preprocessing | Role |
| :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | `n_estimators=150`, `max_depth=10`, `min_samples_split=4`, `min_samples_leaf=2`, `random_state=42` | None | Non-linear ensemble bagging model |
| **Decision Tree Regressor** | `max_depth=6`, `min_samples_split=5`, `random_state=42` | None | Interpretable single-tree baseline |
| **KNN Regressor** | `n_neighbors=7`, `weights="distance"` | `StandardScaler` | Distance-weighted instance-based learner |
| **Ridge Regression** | `alpha=1.0` | `StandardScaler` | L2-regularized linear regression baseline |

---

## 📊 Dataset & Feature Engineering

The application uses **`city_day.csv`** (historical city-level daily air quality observations) containing `City`, `Date`, `PM2.5`, `PM10`, `NO2`, `CO`, `SO2`, `O3`, and `AQI`.

### Engineered Input Features (14 Total)
* **Ambient Pollutants & Index (7):** `PM2.5`, `PM10`, `NO2`, `CO`, `SO2`, `O3`, `AQI`
* **Calendar Features (2):** `Month`, `DayOfWeek`
* **Lagged AQI Features (3):** `AQI_lag1` ($t-1$), `AQI_lag2` ($t-2$), `AQI_lag7` ($t-7$)
* **Rolling Statistics (2):** `AQI_roll3` (3-day rolling mean), `AQI_roll7` (7-day rolling mean)

### Multi-Horizon Targets (3 Total)
* `Target_1D` ($y_{t+1}$), `Target_3D` ($y_{t+3}$), `Target_7D` ($y_{t+7}$)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/aerosense-aqi-forecasting.git
cd aerosense-aqi-forecasting
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv .venv
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
> **Note:** `app.py` requires `matplotlib` for rendering dark-themed charts in addition to the packages in `requirements.txt`.
```bash
pip install -r requirements.txt
pip install matplotlib
```

### 4. Place the Dataset
Ensure `city_day.csv` is located in the root project directory alongside `app.py` (or provide its full file path directly in the dashboard sidebar).

---

## 🖥️ Usage Guide

Start the Streamlit server from the project root directory:
```bash
streamlit run app.py
```

Once launched, open your browser to `http://localhost:8501`:
1. **Configuration Sidebar:** Verify the `city_day.csv` path, select a **City**, and choose an **ML Algorithm**.
2. **What-If Simulator:** Adjust pollutant sliders (`PM2.5`, `PM10`, `NO2`, `CO`, `Current AQI`) or click **↺ Reset Simulator** to restore the latest historical baseline.
3. **Analyze Sections:**
   * **01 / AQI Overview:** View Current, +1 Day, +3 Days, and +7 Days AQI cards with health categories.
   * **02 / Forecast Analysis:** Compare Actual vs. Predicted AQI trajectories and horizon-specific MAE, RMSE, and $R^2$ metrics.
   * **03 / Model Performance:** Inspect the full benchmark table, average MAE algorithm ranking, and grouped metric bar charts.
   * **04 / Model Interpretability:** Explore relative feature importances for `Random Forest` and `Decision Tree`.
   * **05 / Dataset Information:** Review train/test record counts and recent historical records.

---

## 📁 Repository Structure

```text
├── app.py                # Main Streamlit dashboard & multi-horizon ML pipeline
├── city_day.csv          # Historical city-level daily air quality dataset
├── requirements.txt      # Core Python package dependencies
├── AIML Report.pdf       # Trimester-1 academic project report
└── README.md             # Project documentation
```

---

## ⚠️ Limitations & Disclaimer
* **Academic & Educational Prototype:** Predictions and What-If simulator outputs are model-generated estimates based on historical data—they are **not** live sensor readings or certified public-health advisories.
* **Conditional Scenario Modeling:** The feature vector includes same-row pollutant concentrations (`PM2.5`, `PM10`, `NO2`, `CO`, `SO2`, `O3`), making the What-If simulator a conditional scenario estimator rather than a pure forecast using only lagged variables.
* **Exogenous Variables:** Meteorological variables (wind speed, humidity, rainfall, temperature) and traffic data are not present in `city_day.csv`, which limits long-horizon (+3 and +7 day) variance explanation.

---

## 🔮 Future Enhancements
* Integrate meteorological covariates (wind speed, temperature, humidity, precipitation) to improve multi-day accuracy.
* Incorporate gradient boosting (`XGBoost`, `LightGBM`) and sequence deep learning architectures (`LSTM`, `GRU`).
* Connect live air quality APIs (CPCB / OpenAQ) for real-time automated inference and threshold alerts.
* Add geospatial AQI heatmaps across Indian cities.

---

## 🎓 Team & Academic Information

**Course:** `26CS5104E` – AI and Machine Learning with Applied Mathematics  
**Program:** Master of Technology in Artificial Intelligence & Data Science (2026–2027)  
**Institution:** Koneru Lakshmaiah Education Foundation (KL University), Vaddeswaram, Andhra Pradesh  
**Project Supervisor:** Dr. Y. Sankararao, Professor  

### Team 2 Members
| Student ID | Student Name |
| :--- | :--- |
| `2601150026` | Yadla Syam Sundar |
| `2601150034` | Udumula Avinash |
| `2601150039` | Myneni Balavenkata Gokula ShivaRamaKrishna |
| `2601150056` | Maroju Geetha Krishna Venkatesh |
| `2601150060` | Shaik Bhikku Saheb |