import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_battery_health(battery_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(battery_df['cycle_number'], battery_df['capacity'], 'b-o', markersize=3)
    axes[0, 0].set_title('Capacity Fade Over Cycles')
    axes[0, 1].plot(battery_df['cycle_number'], battery_df['soh'], 'g-o', markersize=3)
    axes[0, 1].set_title('SOH Degradation')
    axes[1, 0].plot(battery_df['cycle_number'], battery_df['temp_max'], 'r-o', markersize=3)
    axes[1, 0].set_title('Temperature Profile')
    axes[1, 1].plot(battery_df['cycle_number'], battery_df['voltage_min'], 'purple', marker='o', markersize=3)
    axes[1, 1].set_title('Min Voltage During Discharge')
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names):
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_title('ML Feature Importance')
    return fig
