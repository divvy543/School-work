"""Visualization Utilities"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_battery_health(battery_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(battery_df['cycle_number'], battery_df['capacity'], 'b-o', markersize=3)
    axes[0, 0].axhline(y=1.4, color='r', linestyle='--', label='Failure Threshold (1.4Ah)')
    axes[0, 0].set_xlabel('Cycle Number')
    axes[0, 0].set_ylabel('Capacity (Ah)')
    axes[0, 0].set_title('Capacity Fade Over Cycles')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(battery_df['cycle_number'], battery_df['soh'], 'g-o', markersize=3)
    axes[0, 1].axhline(y=70, color='r', linestyle='--', label='End of Life (70% SOH)')
    axes[0, 1].set_xlabel('Cycle Number')
    axes[0, 1].set_ylabel('State of Health (%)')
    axes[0, 1].set_title('SOH Degradation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(battery_df['cycle_number'], battery_df['temp_max'], 'r-o', markersize=3, label='Max Temp')
    axes[1, 0].plot(battery_df['cycle_number'], battery_df['temp_avg'], 'orange', marker='o', markersize=3, label='Avg Temp')
    axes[1, 0].set_xlabel('Cycle Number')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_title('Temperature Profile')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(battery_df['cycle_number'], battery_df['voltage_min'], 'purple', marker='o', markersize=3)
    axes[1, 1].set_xlabel('Cycle Number')
    axes[1, 1].set_ylabel('Min Voltage (V)')
    axes[1, 1].set_title('Minimum Voltage During Discharge')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names):
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feature_names)))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    ax.set_xlabel('Importance Score')
    ax.set_title('ML Feature Importance')
    return fig
