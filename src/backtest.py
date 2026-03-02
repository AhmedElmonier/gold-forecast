import os
import sys
import pandas as pd
import logging
# Add the parent directory (project root) to sys.path so we can import 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import fetch_all_data, preprocess_data

logger = logging.getLogger(__name__)

def run_backtest() -> str:
    """
    Simulates a trading strategy over the historical data to evaluate performance.
    
    Strategy: 
    - BUY when SMA_20 > SMA_50 (Uptrend) and RSI < 70 (Not Overbought)
    - SELL/FLAT when SMA_20 < SMA_50 (Downtrend) or RSI > 70 (Overbought)
    
    Returns:
        str: A Markdown-formatted string with backtest metrics.
    """
    logger.info("Running backtest simulation...")
    try:
        raw_df = fetch_all_data(period="5y")
        if raw_df.empty:
            return "❌ Error: Failed to fetch historical data for backtesting."
            
        df = preprocess_data(raw_df)
        
        initial_balance = 10000.0
        balance = initial_balance
        position = 0.0  # Amount of Gold held
        buy_price = 0.0
        
        trades = 0
        winning_trades = 0
        peak_balance = initial_balance
        max_drawdown = 0.0
        
        for index, row in df.iterrows():
            current_price = row['y']
            sma_20 = row['SMA_20']
            sma_50 = row['SMA_50']
            rsi = row['RSI_14']
            
            # Daily mark-to-market for drawdown
            current_portfolio_value = balance + (position * current_price)
            if current_portfolio_value > peak_balance:
                peak_balance = current_portfolio_value
            
            drawdown = (peak_balance - current_portfolio_value) / peak_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
            # Trading Logic
            buy_signal = (sma_20 > sma_50) and (rsi < 70)
            sell_signal = (sma_20 < sma_50) or (rsi > 70)
            
            # Execute Trades
            if buy_signal and position == 0:
                # Buy as much as possible
                position = balance / current_price
                balance = 0.0
                buy_price = current_price
                
            elif sell_signal and position > 0:
                # Sell all
                balance = position * current_price
                position = 0.0
                trades += 1
                if current_price > buy_price:
                    winning_trades += 1
                    
        # Final evaluation: close any open positions at the end of the data
        final_price = df.iloc[-1]['y']
        if position > 0:
            balance = position * final_price
            trades += 1
            if final_price > buy_price:
                winning_trades += 1
                
        total_return_pct = ((balance - initial_balance) / initial_balance) * 100
        win_rate = (winning_trades / trades * 100) if trades > 0 else 0.0
        
        # Benchmark approach (Buy and Hold)
        first_price = df.iloc[0]['y']
        buy_and_hold_return = ((final_price - first_price) / first_price) * 100
        
        msg = f"📊 *Backtest Results (Last 5 Years)* 📊\n\n"
        msg += f"• *Strategy:* Trend Following (SMA 20/50 + RSI)\n"
        msg += f"• *Initial Balance:* ${initial_balance:,.2f}\n"
        msg += f"• *Final Balance:* ${balance:,.2f}\n\n"
        
        msg += f"📈 *Performance Metrics*\n"
        msg += f"• *Total Return:* {total_return_pct:.2f}%\n"
        msg += f"• *Buy & Hold Return:* {buy_and_hold_return:.2f}%\n"
        msg += f"• *Max Drawdown:* {max_drawdown * 100:.2f}%\n"
        msg += f"• *Total Trades Executed:* {trades}\n"
        msg += f"• *Win Rate:* {win_rate:.1f}%\n\n"
        
        if total_return_pct > buy_and_hold_return:
            msg += f"🏆 *Verdict:* The AI strategy outperformed holding Gold by {(total_return_pct - buy_and_hold_return):.2f}%!"
        else:
            msg += f"📉 *Verdict:* The AI strategy underperformed a simple Buy & Hold approach."
            
        logger.info("Backtest simulation complete.")
        return msg
        
    except Exception as e:
        logger.error(f"Error during backtest: {e}")
        return f"❌ An error occurred during backtesting: {str(e)}"

if __name__ == "__main__":
    print(run_backtest())
