import sqlite3
import logging
import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DB_NAME = 'portfolio.db'


def init_forecast_history_db():
    """Creates the forecast_history table in portfolio.db if it doesn't exist."""
    logger.info("Initializing Forecast History table...")
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS forecast_history (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    forecast_date     TEXT NOT NULL,
                    ticker            TEXT NOT NULL,
                    days_ahead        INTEGER NOT NULL,
                    target_date       TEXT NOT NULL,
                    predicted_price   REAL NOT NULL,
                    prophet_price     REAL,
                    xgb_price         REAL,
                    price_at_forecast REAL NOT NULL,
                    actual_price      REAL,
                    abs_error         REAL,
                    pct_error         REAL,
                    direction_correct INTEGER
                )
            ''')
            conn.commit()
        logger.info("Forecast History table ready.")
    except Exception as e:
        logger.error(f"Failed to initialize forecast_history table: {e}")


def save_forecast(
    ticker: str,
    days_ahead: int,
    predicted_price: float,
    price_at_forecast: float,
    prophet_price: Optional[float] = None,
    xgb_price: Optional[float] = None,
) -> bool:
    """
    Saves a new forecast entry to the forecast_history table.

    Args:
        ticker: The asset ticker (e.g. 'GC=F').
        days_ahead: Forecast horizon in days.
        predicted_price: The final ensemble predicted price.
        price_at_forecast: The actual market price at the time of forecasting.
        prophet_price: Raw Prophet prediction (before ensembling).
        xgb_price: Raw XGBoost prediction (before ensembling).

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        today = datetime.date.today()
        target_date = today + datetime.timedelta(days=days_ahead)

        with sqlite3.connect(DB_NAME) as conn:
            conn.execute('''
                INSERT INTO forecast_history
                    (forecast_date, ticker, days_ahead, target_date,
                     predicted_price, prophet_price, xgb_price, price_at_forecast)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                today.isoformat(),
                ticker,
                days_ahead,
                target_date.isoformat(),
                predicted_price,
                prophet_price,
                xgb_price,
                price_at_forecast,
            ))
            conn.commit()

        logger.info(
            f"Saved forecast: {ticker} predicted=${predicted_price:.2f} "
            f"(from ${price_at_forecast:.2f}) targeting {target_date.isoformat()}"
        )
        return True

    except Exception as e:
        logger.error(f"Error saving forecast to history: {e}")
        return False


def reconcile_forecasts():
    """
    For every unresolved forecast whose target_date has arrived or passed,
    fetches the actual price from yfinance and fills in actual_price,
    abs_error, pct_error, and direction_correct.
    """
    import yfinance as yf

    today = datetime.date.today().isoformat()

    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT id, ticker, days_ahead, target_date,
                       predicted_price, price_at_forecast
                FROM forecast_history
                WHERE actual_price IS NULL
                  AND target_date <= ?
            ''', (today,)).fetchall()

        if not rows:
            logger.info("No past forecasts to reconcile.")
            return

        logger.info(f"Reconciling {len(rows)} past forecast(s)...")

        for row in rows:
            row_id = row['id']
            ticker = row['ticker']
            target_date_str = row['target_date']
            predicted_price = row['predicted_price']
            price_at_forecast = row['price_at_forecast']

            try:
                # Fetch a short window of data around the target date to get
                # the closest available closing price (weekends/holidays
                # may mean no exact match).
                target_dt = datetime.date.fromisoformat(target_date_str)
                start = target_dt - datetime.timedelta(days=5)
                end = target_dt + datetime.timedelta(days=5)

                hist = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    progress=False,
                    auto_adjust=True,
                )

                if hist.empty:
                    logger.warning(f"No price data found for {ticker} around {target_date_str}. Skipping.")
                    continue

                # Pick the closest trading day to target_date using a simple
                # day-delta comparison — avoids all timezone / TimedeltaIndex issues.
                trading_dates = [d.date() for d in hist.index]
                closest_idx = min(
                    range(len(trading_dates)),
                    key=lambda i: abs((trading_dates[i] - target_dt).days)
                )

                # yfinance may return MultiIndex columns; flatten to a scalar safely
                close_col = hist['Close']
                if hasattr(close_col, 'iloc'):
                    close_val = close_col.iloc[closest_idx]
                    # If it's still a Series (MultiIndex case), take the first element
                    if hasattr(close_val, 'iloc'):
                        close_val = close_val.iloc[0]
                    actual_price = float(close_val)
                else:
                    actual_price = float(close_col[closest_idx])

                abs_error = abs(actual_price - predicted_price)
                pct_error = (abs_error / actual_price) * 100

                # Direction correct: did we predict the right direction vs price at forecast?
                predicted_direction = predicted_price >= price_at_forecast
                actual_direction = actual_price >= price_at_forecast
                direction_correct = 1 if predicted_direction == actual_direction else 0

                with sqlite3.connect(DB_NAME) as conn:
                    conn.execute('''
                        UPDATE forecast_history
                        SET actual_price      = ?,
                            abs_error         = ?,
                            pct_error         = ?,
                            direction_correct = ?
                        WHERE id = ?
                    ''', (actual_price, abs_error, pct_error, direction_correct, row_id))
                    conn.commit()

                logger.info(
                    f"Reconciled [{row_id}] {ticker} target={target_date_str}: "
                    f"predicted=${predicted_price:.2f}, actual=${actual_price:.2f}, "
                    f"err={abs_error:.2f} ({pct_error:.2f}%), "
                    f"direction={'✅' if direction_correct else '❌'}"
                )

            except Exception as e:
                logger.error(f"Error reconciling row {row_id}: {e}")

    except Exception as e:
        logger.error(f"Error during reconcile_forecasts: {e}")


def get_accuracy_stats(ticker: str = "GC=F", limit: int = 30) -> dict:
    """
    Returns summary accuracy statistics and the most recent reconciled forecasts.

    Args:
        ticker: Asset to filter by.
        limit: How many reconciled rows to consider for stats.

    Returns:
        dict with keys: total_forecasts, reconciled_count, mae, mape,
                        direction_hit_rate, recent_forecasts (list of dicts)
    """
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row

            total = conn.execute(
                "SELECT COUNT(*) FROM forecast_history WHERE ticker = ?", (ticker,)
            ).fetchone()[0]

            reconciled_rows = conn.execute('''
                SELECT forecast_date, target_date, predicted_price,
                       price_at_forecast, actual_price,
                       abs_error, pct_error, direction_correct
                FROM forecast_history
                WHERE ticker = ? AND actual_price IS NOT NULL
                ORDER BY target_date DESC
                LIMIT ?
            ''', (ticker, limit)).fetchall()

        if not reconciled_rows:
            return {
                "total_forecasts": total,
                "reconciled_count": 0,
                "mae": None,
                "mape": None,
                "direction_hit_rate": None,
                "recent_forecasts": [],
            }

        mae = sum(r['abs_error'] for r in reconciled_rows) / len(reconciled_rows)
        mape = sum(r['pct_error'] for r in reconciled_rows) / len(reconciled_rows)
        directions = [r['direction_correct'] for r in reconciled_rows]
        direction_hit_rate = (sum(directions) / len(directions)) * 100

        recent = [dict(r) for r in reconciled_rows[:5]]

        return {
            "total_forecasts": total,
            "reconciled_count": len(reconciled_rows),
            "mae": mae,
            "mape": mape,
            "direction_hit_rate": direction_hit_rate,
            "recent_forecasts": recent,
        }

    except Exception as e:
        logger.error(f"Error getting accuracy stats: {e}")
        return {
            "total_forecasts": 0,
            "reconciled_count": 0,
            "mae": None,
            "mape": None,
            "direction_hit_rate": None,
            "recent_forecasts": [],
        }


# Initialize the table whenever this module is imported
init_forecast_history_db()
