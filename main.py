import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import requests, zipfile, io, os, sys
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.data_parser import parse_nasa_battery, build_nasa_dataset
from utils.visualization import plot_battery_health, plot_feature_importance

st.set_page_config(page_title="NASA Battery AI", layout="wide")
st.title("  NASA Battery Dataset + ML Simulator")

@st.cache_data
def download_nasa_data():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)
    target_file = os.path.join(data_dir, "B0005.mat")
    if not os.path.exists(target_file):
        url = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
        resp = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z: z.extractall(data_dir)
    return data_dir

data_dir = download_nasa_data()
df = build_nasa_dataset(data_dir)

st.header("  NASA Battery Data Overview")
battery_id = st.sidebar.selectbox("Select Battery", df['battery_id'].unique())
b_df = df[df['battery_id'] == battery_id]
st.pyplot(plot_battery_health(b_df))

# Sidebar ML logic and simulation code goes here...
