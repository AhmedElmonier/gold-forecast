import sqlite3
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

DB_NAME = 'portfolio.db'

def init_db():
    """Initializes the SQLite database with users and portfolios tables."""
    logger.info("Initializing Portfolio SQLite Database...")
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            
            # Create users table to track USD balances
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    usd_balance REAL DEFAULT 100000.0
                )
            ''')
            
            # Create portfolios table to track asset holdings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolios (
                    user_id INTEGER,
                    ticker TEXT,
                    quantity REAL,
                    avg_price REAL,
                    PRIMARY KEY (user_id, ticker),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to init database: {e}")

def get_balance(user_id: int) -> float:
    """Gets the available USD balance for a user. Creates the user if they don't exist."""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT usd_balance FROM users WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            
            if result:
                return result[0]
            else:
                # User doesn't exist, create them with default balance
                cursor.execute("INSERT INTO users (user_id, usd_balance) VALUES (?, ?)", (user_id, 100000.0))
                conn.commit()
                return 100000.0
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        return 0.0

def get_portfolio(user_id: int) -> Dict[str, Dict[str, float]]:
    """Retrieves the user's current holdings."""
    holdings = {}
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ticker, quantity, avg_price FROM portfolios WHERE user_id = ? AND quantity > 0", (user_id,))
            rows = cursor.fetchall()
            
            for row in rows:
                ticker, quantity, avg_price = row
                holdings[ticker] = {
                    "quantity": quantity,
                    "avg_price": avg_price
                }
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
    return holdings

def execute_trade(user_id: int, ticker: str, quantity: float, side: str, current_price: float) -> Tuple[bool, str]:
    """
    Executes a virtual trade.
    side: 'BUY' or 'SELL'
    Returns a boolean success flag and a string message.
    """
    ticker = ticker.upper()
    side = side.upper()
    if quantity <= 0:
        return False, "Quantity must be greater than zero."
        
    balance = get_balance(user_id)
    holdings = get_portfolio(user_id)
    total_cost = quantity * current_price
    
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            
            if side == 'BUY':
                if balance < total_cost:
                    return False, f"Insufficient funds. Required: ${total_cost:.2f}, Available: ${balance:.2f}"
                
                # Deduct from balance
                new_balance = balance - total_cost
                cursor.execute("UPDATE users SET usd_balance = ? WHERE user_id = ?", (new_balance, user_id))
                
                # Update holdings
                if ticker in holdings:
                    old_qty = holdings[ticker]['quantity']
                    old_avg = holdings[ticker]['avg_price']
                    
                    new_qty = old_qty + quantity
                    # Calculate new weighted average price
                    new_avg = ((old_qty * old_avg) + total_cost) / new_qty
                    
                    cursor.execute("UPDATE portfolios SET quantity = ?, avg_price = ? WHERE user_id = ? AND ticker = ?", 
                                   (new_qty, new_avg, user_id, ticker))
                else:
                    cursor.execute("INSERT INTO portfolios (user_id, ticker, quantity, avg_price) VALUES (?, ?, ?, ?)",
                                   (user_id, ticker, quantity, current_price))
                                   
                conn.commit()
                return True, f"✅ Successfully purchased {quantity} {ticker} at ${current_price:.2f}."
                
            elif side == 'SELL':
                if ticker not in holdings or holdings[ticker]['quantity'] < quantity:
                    return False, f"Insufficient {ticker} balance. You own {holdings.get(ticker, {}).get('quantity', 0)}."
                
                # Add to balance
                new_balance = balance + total_cost
                cursor.execute("UPDATE users SET usd_balance = ? WHERE user_id = ?", (new_balance, user_id))
                
                # Update holdings
                new_qty = holdings[ticker]['quantity'] - quantity
                
                if new_qty == 0:
                    cursor.execute("DELETE FROM portfolios WHERE user_id = ? AND ticker = ?", (user_id, ticker))
                else:
                    cursor.execute("UPDATE portfolios SET quantity = ? WHERE user_id = ? AND ticker = ?", 
                                   (new_qty, user_id, ticker))
                                   
                conn.commit()
                
                # Calculate profit/loss
                avg_buy_price = holdings[ticker]['avg_price']
                pl_per_unit = current_price - avg_buy_price
                total_pl = pl_per_unit * quantity
                pl_str = f"🚀 Profit: ${total_pl:.2f}" if total_pl >= 0 else f"🩸 Loss: ${total_pl:.2f}"
                
                return True, f"✅ Successfully sold {quantity} {ticker} at ${current_price:.2f}.\n{pl_str}"
            
            else:
                return False, "Invalid trade side. Must be 'BUY' or 'SELL'."
                
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False, f"System error processing trade: {e}"

# Run initialization when imported
init_db()
