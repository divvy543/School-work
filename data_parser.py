import numpy as np
import pandas as pd
import os
from scipy.io import loadmat

def parse_nasa_battery(filepath):
    mat_data = loadmat(filepath)
    battery_name = [k for k in mat_data.keys() if not k.startswith('__')][0]
    battery = mat_data[battery_name]
    cycles = []
    for cycle in battery[0][0]['cycle'][0]:
        if cycle['type'][0] == 'discharge':
            data = cycle['data']
            cycles.append({
                'capacity': data[0][0]['Capacity'][0][0] if 'Capacity' in data[0][0].dtype.names else None,
                'voltage_min': np.min(data[0][0]['Voltage_measured'][0]),
                'temp_max': np.max(data[0][0]['Temperature_measured'][0]),
                'current_avg': np.mean(np.abs(data[0][0]['Current_measured'][0])),
                'duration': data[0][0]['Time'][0][-1] - data[0][0]['Time'][0][0]
            })
    return cycles

def build_nasa_dataset(data_dir):
    all_data = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    for f in files:
        path = os.path.join(data_dir, f)
        cycles = parse_nasa_battery(path)
        for i, c in enumerate([cy for cy in cycles if cy['capacity'] is not None]):
            all_data.append({'battery_id': f.split('.')[0], 'cycle_number': i, **c})
    df = pd.DataFrame(all_data)
    if not df.empty:
        for b in df['battery_id'].unique():
            mask = df['battery_id'] == b
            df.loc[mask, 'soh'] = (df.loc[mask, 'capacity'] / df.loc[mask, 'capacity'].iloc[0]) * 100
    return df
