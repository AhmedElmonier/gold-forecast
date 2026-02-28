import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def generate_forecast_chart(historical_data: pd.DataFrame, forecast_data: pd.DataFrame, filename: str = "forecast_chart.png") -> str:
    """
    Generates a visual chart combining historical prices, SMAs, and the future forecast.
    Saves it to the disk and returns the filepath.
    """
    logger.info("Generating visual forecast chart...")
    
    # We only want to plot the last ~180 days to keep the chart readable
    hist_subset = historical_data.tail(180)
    
    # Only plot the days in the forecast that are IN THE FUTURE
    future_forecast = forecast_data[forecast_data['ds'] > hist_subset['ds'].max()]
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="darkgrid")
    
    # Plot historical Close prices
    plt.plot(hist_subset['ds'], hist_subset['y'], label='Historical Close Price', color='white', linewidth=2)
    
    # Plot SMAs
    plt.plot(hist_subset['ds'], hist_subset['SMA_20'], label='20-Day SMA', color='cyan', linestyle='--', linewidth=1.5)
    plt.plot(hist_subset['ds'], hist_subset['SMA_50'], label='50-Day SMA', color='magenta', linestyle='--', linewidth=1.5)
    
    # Plot the Forecast line
    plt.plot(future_forecast['ds'], future_forecast['yhat'], label='Prophet Forecast', color='yellow', linewidth=2)
    
    # Shade the Confidence Interval
    plt.fill_between(
        future_forecast['ds'], 
        future_forecast['yhat_lower'], 
        future_forecast['yhat_upper'], 
        color='gold', alpha=0.3, label='Confidence Interval'
    )
    
    plt.title('Gold Price Forecast (GC=F)', fontsize=16, color='white', pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='upper left', frameon=True, facecolor='black', edgecolor='gray', labelcolor='white')
    
    # Style tweaks for visibility
    ax = plt.gca()
    ax.set_facecolor('#1e1e2e')  # Dark background
    plt.gcf().patch.set_facecolor('#1e1e2e')
    ax.tick_params(colors='lightgray')
    for spine in ax.spines.values():
        spine.set_color('gray')

    plt.tight_layout()
    
    filepath = os.path.join(os.getcwd(), filename)
    plt.savefig(filepath, dpi=150, facecolor='#1e1e2e')
    plt.close()
    
    logger.info(f"Chart saved to {filepath}")
    return filepath
