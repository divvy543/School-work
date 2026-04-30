import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import requests
import zipfile
import io
import os
import sys
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.data_parser import parse_nasa_battery, build_nasa_dataset
from utils.visualization import plot_battery_health, plot_feature_importance

NASA_BATTERY_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

st.set_page_config(page_title="NASA Battery AI", layout="wide")
st.title("🔋 NASA Battery Dataset + ML Simulator")
st.markdown("*Real data from NASA Ames Research Center*")

@st.cache_data
def download_nasa_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(os.path.join(DATA_DIR, "B0005.mat")):
        st.success("✅ NASA data already cached!")
    else:
        with st.spinner("Downloading NASA Battery Dataset (~2MB)..."):
            try:
                response = requests.get(NASA_BATTERY_URL, timeout=120)
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall(DATA_DIR)
                st.success("✅ Download complete!")
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None
    return DATA_DIR

st.sidebar.header("🚀 Data Source")
data_dir = download_nasa_data()
df = build_nasa_dataset(data_dir)

st.header("📊 NASA Battery Data Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Batteries", df['battery_id'].nunique())
col2.metric("Total Cycles", len(df))
col3.metric("Capacity Range", f"{df['capacity'].min():.2f} - {df['capacity'].max():.2f} Ah")

selected_battery = st.sidebar.selectbox("Select Battery", df['battery_id'].unique())
battery_df = df[df['battery_id'] == selected_battery].copy()

st.subheader(f"Real Degradation: {selected_battery}")
fig = plot_battery_health(battery_df)
st.pyplot(fig)

st.header("🤖 Machine Learning: Predict Battery Health")
st.sidebar.header("⚙️ ML Settings")

features = ['cycle_number', 'temp_max', 'temp_avg', 'current_avg', 'voltage_min', 'duration']
target = 'soh'

X = df[features]
y = df[target]

test_size = st.sidebar.slider("Test Size %", 10, 40, 20) / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

n_estimators = st.sidebar.slider("Trees in Forest", 50, 300, 100)
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

col1, col2 = st.columns(2)
col1.metric("Model Accuracy (R²)", f"{r2:.3f}")
col2.metric("Prediction Error (RMSE)", f"{rmse:.2f}%")

st.subheader("What Affects Battery Health Most?")
fig_imp = plot_feature_importance(model, features)
st.pyplot(fig_imp)

st.header("🔮 Predict Your Battery's Future")
st.markdown("Use the sliders to simulate different conditions and predict SOH:")

col1, col2, col3 = st.columns(3)
with col1:
    pred_cycles = st.slider("Future Cycles", 0, 500, 100)
    pred_temp_max = st.slider("Max Temperature (°C)", 20, 60, 35)
with col2:
    pred_temp_avg = st.slider("Avg Temperature (°C)", 15, 50, 25)
    pred_current = st.slider("Current Draw (A)", 0.5, 5.0, 2.0)
with col3:
    pred_voltage = st.slider("Min Voltage (V)", 2.0, 3.5, 2.7)
    pred_duration = st.slider("Cycle Duration (s)", 500, 3000, 1500)

future_data = pd.DataFrame([{
    'cycle_number': pred_cycles,
    'temp_max': pred_temp_max,
    'temp_avg': pred_temp_avg,
    'current_avg': pred_current,
    'voltage_min': pred_voltage,
    'duration': pred_duration
}])

predicted_soh = model.predict(future_data)[0]
predicted_soh = np.clip(predicted_soh, 50, 100)

st.metric("Predicted State of Health", f"{predicted_soh:.1f}%")

if predicted_soh < 70:
    st.error("🔴 CRITICAL: Battery at End of Life! Replace immediately.")
elif predicted_soh < 80:
    st.warning("🟡 WARNING: Significant degradation. Plan replacement soon.")
elif predicted_soh < 90:
    st.info("🟢 CAUTION: Moderate wear. Monitor closely.")
else:
    st.success("✅ HEALTHY: Battery in good condition!")

st.header("🔬 Real-Time Discharge Simulation (ML-Enhanced)")
st.sidebar.header("🔋 Simulation")
sim_hours = st.sidebar.slider("Sim Hours", 0.5, 5.0, 2.0)
load_current = st.sidebar.slider("Load (A)", 0.1, 5.0, 2.0)

time_steps = int(sim_hours * 60)
time = np.linspace(0, sim_hours, time_steps)
real_capacity = battery_df['capacity'].iloc[0] if len(battery_df) > 0 else 2.0

# Simple BMS
soc_simple = np.ones(time_steps) * 100
v_simple = np.ones(time_steps) * 3.7
for i in range(1, time_steps):
    drain = (load_current / real_capacity / 60) * 100
    soc_simple[i] = max(soc_simple[i-1] - drain, 0)
    v_simple[i] = 3.7 * (soc_simple[i] / 100)

# Smart BMS
soc_smart = np.ones(time_steps) * 100
v_smart = np.ones(time_steps) * 3.7

if len(battery_df) > 10:
    degradation_rate = (battery_df['soh'].iloc[0] - battery_df['soh'].iloc[-1]) / len(battery_df)
else:
    degradation_rate = 0.1

k = 0.3 + (degradation_rate / 10)

for i in range(1, time_steps):
    health_penalty = 1 + (100 - predicted_soh) / 200
    low_soc_penalty = 1 + k * (1 - soc_smart[i-1]/100)
    drain = (load_current / real_capacity / 60) * 100 * health_penalty * low_soc_penalty
    soc_smart[i] = max(soc_smart[i-1] - drain, 0)
    v_smart[i] = 3.7 * (1 - k * (1 - soc_smart[i]/100)**2)

fig_sim, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax1.plot(time, soc_simple/100, 'r--', label='Simple BMS', linewidth=2, alpha=0.7)
ax1.plot(time, soc_smart/100, 'b-', label='ML-Enhanced BMS (NASA-trained)', linewidth=2)
ax1.axhline(y=0.2, color='orange', linestyle=':', alpha=0.7, label='Critical Low')
ax1.set_ylabel('State of Charge')
ax1.set_ylim(0, 1.1)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('Battery % Over Time')

ax2.plot(time, v_simple, 'r--', linewidth=2, alpha=0.7)
ax2.plot(time, v_smart, 'b-', linewidth=2)
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Cell Voltage (V)')
ax2.set_title('Voltage Drop')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig_sim)

st.download_button(
    "Download NASA Data + Predictions (CSV)",
    df.to_csv(index=False),
    "nasa_battery_data.csv",
    "text/csv"
)

st.markdown("---")
st.caption("Data: NASA Prognostics Center of Excellence | Model: Random Forest Regressor | Built with Streamlit")
