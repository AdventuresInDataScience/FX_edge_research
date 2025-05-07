#%% Imports
import pandas as pd
import numpy as np
from datetime import datetime as dt
from mango import scheduler, Tuner
data_path = "C:/Users/malha/Documents/Data/FX/EURUSD_H1.csv"
#%% Read data
df = pd.read_csv(data_path, sep="\t", header=1, parse_dates=[0], index_col=0)
df = df.iloc[:, :5]
# Training data, at 75% of the data
train_size = int(len(df) * 0.75)
train = df.iloc[:train_size].copy().reset_index()
# Test data, at 25% of the data
test = df.iloc[train_size:].copy().reset_index()


#%% backest strategy function and metric functions
def single_backtest(df, direction, risk_percentage, spread, 
                      datetime_col='Datetime', open_col='Open', high_col='High', 
                      low_col='Low', close_col='Close', volume_col='Volume',
                      open_signal_col='OpenSignal', close_signal_col='CloseSignal',
                      stop_loss=None, take_profit=None, 
                      pip_value=0.0001):
    """
    # This function backtests a simple trading strategy with stop loss and take profit.
    # It uses numpy for faster calculations and can handle both buy and sell trades.
    # The function takes a dataframe with OHLCV data and calculates the balance and equity over time.
    # It also allows for customization of the stop loss, take profit, and risk percentage.
    # The function returns a dataframe with the datetime, balance, and equity columns.
    Backtests a trading strategy with open/close signals and optional SL/TP.
    Signals are expected to be in the dataframe as 1 for open/close and 0 for no signal.
    The Signals are expected to be in the same row as the OHLCV data. Therefore, signals generated
    from the close of the previous bar will be used to open/close trades on the current bar.
    (they may thus need 'shifting down' to be in the same row as the OHLCV data).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with OHLCV data and signal columns
    direction : str
        'buy' or 'sell'
    risk_percentage : float
        Percentage of balance to risk (0-100)
    spread : float
        Spread in pips
    datetime_col, open_col, high_col, low_col, close_col, volume_col : str
        Column names for OHLCV data
    open_signal_col : str
        Column name for open trade signals (1 for signal, 0 for no signal)
    close_signal_col : str
        Column name for close trade signals (1 for signal, 0 for no signal)
    stop_loss : float or None
        Stop loss in pips (None to ignore stop loss)
    take_profit : float or None
        Take profit in pips (None to ignore take profit)
    pip_value : float
        Value of one pip in price terms (default: 0.0001 for most forex pairs)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with datetime, balance, equity, and trade result columns
    """
    # Convert to numpy for faster processing
    datetime_array = df[datetime_col].values
    open_array = df[open_col].values.astype(np.float64)
    high_array = df[high_col].values.astype(np.float64)
    low_array = df[low_col].values.astype(np.float64)
    close_array = df[close_col].values.astype(np.float64)
    open_signal_array = df[open_signal_col].values.astype(np.int32)
    close_signal_array = df[close_signal_col].values.astype(np.int32)
    
    # Initialize arrays to store results
    n = len(df)
    balance_array = np.ones(n)  # Starting balance of 1
    equity_array = np.ones(n)   # Starting equity of 1
    trade_result_array = np.zeros(n)  # 0=no trade, 1=win, -1=loss
    
    # Initialize trading variables
    current_balance = 1.0
    current_equity = 1.0
    trade_open = False
    entry_price = 0.0
    stop_loss_price = None
    take_profit_price = None
    position_size = 0.0
    actual_entry_price = 0.0
    
    # Loop through each row
    for i in range(n):
        # If no trade is open, check for open signal
        if not trade_open and open_signal_array[i] == 1:
            # Don't open a trade if there's also a close signal (conflicting signals)
            if close_signal_array[i] == 1:
                pass  # Ignore this trade opportunity
            else:
                # Open a trade at the open price of current bar
                entry_price = open_array[i]
                
                # Set stop loss and take profit prices if they're provided
                if stop_loss is not None:
                    if direction.lower() == 'buy':
                        stop_loss_price = entry_price - (stop_loss * pip_value)
                    else:  # sell
                        stop_loss_price = entry_price + (stop_loss * pip_value)
                
                if take_profit is not None:
                    if direction.lower() == 'buy':
                        take_profit_price = entry_price + (take_profit * pip_value)
                    else:  # sell
                        take_profit_price = entry_price - (take_profit * pip_value)
                
                # Adjust entry price for spread
                if direction.lower() == 'buy':
                    # Buy at ask, which is higher
                    actual_entry_price = entry_price + (spread * pip_value)
                else:  # sell
                    # Sell at bid, which is lower
                    actual_entry_price = entry_price - (spread * pip_value)
                
                # Calculate position size based on risk percentage
                if stop_loss is not None:
                    risk_amount = current_balance * (risk_percentage / 100)
                    pip_risk = stop_loss * pip_value  # Convert pips to price
                    position_size = risk_amount / pip_risk if pip_risk > 0 else 0
                else:
                    # Without stop loss, risk a fixed percentage of balance
                    risk_amount = current_balance * (risk_percentage / 100)
                    default_pip_risk = 100 * pip_value
                    position_size = risk_amount / default_pip_risk
                
                trade_open = True
        
        # If a trade is open, check if it should be closed
        if trade_open:
            current_close = close_array[i]
            current_high = high_array[i]
            current_low = low_array[i]
            
            # Initialize flags for closing conditions
            close_by_signal = False
            close_by_sl = False
            close_by_tp = False
            
            # Check if close signal is triggered
            if close_signal_array[i] == 1:
                close_by_signal = True
            
            # Check if stop loss is hit (if stop_loss is provided)
            if stop_loss_price is not None:
                if direction.lower() == 'buy' and current_low <= stop_loss_price:
                    close_by_sl = True
                elif direction.lower() == 'sell' and current_high >= stop_loss_price:
                    close_by_sl = True
            
            # Check if take profit is hit (if take_profit is provided)
            if take_profit_price is not None:
                if direction.lower() == 'buy' and current_high >= take_profit_price:
                    close_by_tp = True
                elif direction.lower() == 'sell' and current_low <= take_profit_price:
                    close_by_tp = True
            
            # Determine closing price and calculate PnL
            pnl = 0
            if close_by_sl and (close_by_tp or close_by_signal):
                # If both SL and TP/signal are hit, SL takes precedence
                if direction.lower() == 'buy':
                    pnl = (stop_loss_price - actual_entry_price) * position_size
                else:  # sell
                    pnl = (actual_entry_price - stop_loss_price) * position_size
                
                current_balance += pnl
                current_equity = current_balance
                trade_open = False
                
                # Record trade result (always a loss with SL)
                trade_result_array[i] = -1
                
            elif close_by_sl:
                # Close at stop loss
                if direction.lower() == 'buy':
                    pnl = (stop_loss_price - actual_entry_price) * position_size
                else:  # sell
                    pnl = (actual_entry_price - stop_loss_price) * position_size
                
                current_balance += pnl
                current_equity = current_balance
                trade_open = False
                
                # Record trade result (always a loss with SL)
                trade_result_array[i] = -1
                
            elif close_by_tp:
                # Close at take profit
                if direction.lower() == 'buy':
                    pnl = (take_profit_price - actual_entry_price) * position_size
                else:  # sell
                    pnl = (actual_entry_price - take_profit_price) * position_size
                
                current_balance += pnl
                current_equity = current_balance
                trade_open = False
                
                # Record trade result (always a win with TP)
                trade_result_array[i] = 1
                
            elif close_by_signal:
                # Close at current close price
                if direction.lower() == 'buy':
                    pnl = (current_close - actual_entry_price) * position_size
                else:  # sell
                    pnl = (actual_entry_price - current_close) * position_size
                
                current_balance += pnl
                current_equity = current_balance
                trade_open = False
                
                # Record trade result based on PnL
                trade_result_array[i] = 1 if pnl > 0 else -1
                
            else:
                # Trade remains open, update equity based on current close
                if direction.lower() == 'buy':
                    unrealized_pnl = (current_close - actual_entry_price) * position_size
                else:  # sell
                    unrealized_pnl = (actual_entry_price - current_close) * position_size
                
                current_equity = current_balance + unrealized_pnl
                
                # If this is the last candle, close the trade
                if i == n - 1:
                    current_balance += unrealized_pnl
                    current_equity = current_balance
                    trade_open = False
                    
                    # Record trade result based on PnL
                    trade_result_array[i] = 1 if unrealized_pnl > 0 else -1
        
        # Store current balance and equity for this row
        balance_array[i] = current_balance
        equity_array[i] = current_equity
    
    # Create and return the results dataframe
    result_df = pd.DataFrame({
        'Datetime': datetime_array,
        'Balance': balance_array,
        'Equity': equity_array,
        'TradeResult': trade_result_array
    })
    
    # Add phantom row with balance and equity of 1.0 at the beginning
    if len(result_df) > 0:
        # Create a phantom datetime just before the first entry
        if hasattr(result_df['Datetime'][0], 'to_pydatetime'):
            phantom_datetime = result_df['Datetime'][0].to_pydatetime() - pd.Timedelta(minutes=1)
        elif isinstance(result_df['Datetime'][0], pd.Timestamp):
            phantom_datetime = result_df['Datetime'][0] - pd.Timedelta(minutes=1)
        elif isinstance(result_df['Datetime'][0], dt):
            phantom_datetime = result_df['Datetime'][0] - pd.Timedelta(minutes=1)
        elif isinstance(result_df['Datetime'][0], str):
            phantom_datetime = pd.to_datetime(result_df['Datetime'][0]) - pd.Timedelta(minutes=1)
        else:
            phantom_datetime = "Initial"
            
        phantom_row = pd.DataFrame({
            'Datetime': [phantom_datetime],
            'Balance': [1.0],
            'Equity': [1.0],
            'TradeResult': [0]
        })
        
        # Concatenate the phantom row at the beginning
        result_df = pd.concat([phantom_row, result_df], ignore_index=True)
    
    return result_df

def calculate_trade_metrics(backtest_df, metric='all'):
    """
    Calculate trading metrics from backtest results.
    
    Parameters:
    -----------
    backtest_df : pandas.DataFrame
        Dataframe returned from backtest_strategy function
    metric : str
        Specific metric to calculate. Options:
        - 'final_equity': Final equity value
        - 'total_trades': Total number of trades taken
        - 'winning_trades': Number of winning trades
        - 'losing_trades': Number of losing trades
        - 'win_rate': Percentage of winning trades
        - 'profit_factor': Gross profits divided by gross losses
        - 'max_consecutive_wins': Maximum consecutive winning trades
        - 'max_consecutive_losses': Maximum consecutive losing trades
        - 'total_return': Total return percentage
        - 'cagr': Compound Annual Growth Rate
        - 'sharpe': Sharpe ratio (using log returns)
        - 'max_drawdown': Maximum drawdown percentage
        - 'ulcer_index': Ulcer index (root mean square of drawdowns)
        - 'std_dev': Annualized standard deviation of log returns
        - 'cagr_mdd_ratio': CAGR to Max Drawdown ratio
        - 'time_in_market': Percentage of time with open positions
        - 'all': Returns all metrics as a dictionary
        
    Returns:
    --------
    float or dict
        Requested metric value or dictionary of all metrics
    """
    # For minimal computation when only one metric is needed
    if metric == 'final_equity':
        return backtest_df['Equity'].iloc[-1]
    
    # Basic metrics from the backtest DataFrame
    final_equity = backtest_df['Equity'].iloc[-1]
    initial_balance = backtest_df['Balance'].iloc[0]
    n_periods = len(backtest_df)
    
    # For CAGR calculation only
    if metric == 'cagr':
        if n_periods > 1:
            # Assuming 252 trading days per year - adjust based on your timeframe
            timeframe_factor = 6000  # Adjust based on timeframe
            return (final_equity / initial_balance) ** (timeframe_factor / n_periods) - 1
        else:
            return 0    
    
    # For total return calculation only
    if metric == 'total_return':
        return (final_equity / initial_balance) - 1 if initial_balance > 0 else 0
    
    # For trade count metrics, we need to analyze TradeResult
    trade_results = backtest_df['TradeResult'].values
    winning_trades = (trade_results == 1).sum()
    losing_trades = (trade_results == -1).sum()
    total_trades = winning_trades + losing_trades
    
    # Return specific count metrics
    if metric == 'total_trades':
        return total_trades
    elif metric == 'winning_trades':
        return winning_trades
    elif metric == 'losing_trades':
        return losing_trades
    
    # For win rate calculation
    if metric == 'win_rate':
        return winning_trades / total_trades if total_trades > 0 else 0
    
    # For profit factor calculation
    if metric == 'profit_factor':
        pnl_series = backtest_df['Balance'].diff().fillna(0)
        profits = pnl_series[pnl_series > 0].sum()
        losses = abs(pnl_series[pnl_series < 0].sum())
        return profits / losses if losses != 0 else float('inf')
    
    # For consecutive trades metrics
    if metric == 'max_consecutive_wins' or metric == 'max_consecutive_losses':
        # Filter out non-trade periods
        trade_results_no_zero = [r for r in trade_results if r != 0]
        
        consecutive_wins = 0
        current_win_streak = 0
        consecutive_losses = 0
        current_loss_streak = 0
        
        for result in trade_results_no_zero:
            if result == 1:  # Win
                current_win_streak += 1
                current_loss_streak = 0
                consecutive_wins = max(consecutive_wins, current_win_streak)
            elif result == -1:  # Loss
                current_loss_streak += 1
                current_win_streak = 0
                consecutive_losses = max(consecutive_losses, current_loss_streak)
        
        if metric == 'max_consecutive_wins':
            return consecutive_wins
        else:  # max_consecutive_losses
            return consecutive_losses
    
    # For Sharpe ratio calculation and standard deviation
    log_returns = None
    annualized_std = None
    if metric in ['sharpe', 'std_dev']:
        if n_periods > 1:
            # Calculate log returns
            log_returns = np.diff(np.log(backtest_df['Equity'].values))
            # Annualize standard deviation
            timeframe_factor = 252  # Adjust based on timeframe
            annualized_std = np.std(log_returns) * np.sqrt(timeframe_factor)
            
            if metric == 'std_dev':
                return annualized_std
            
            # For Sharpe, continue with calculation
            annualized_mean = np.mean(log_returns) * timeframe_factor
            # Calculate Sharpe ratio
            return annualized_mean / annualized_std if annualized_std > 0 else 0
        else:
            return 0
    
    # For maximum drawdown calculation
    if metric in ['max_drawdown', 'cagr_mdd_ratio', 'ulcer_index']:
        equity_series = backtest_df['Equity'].values
        peak = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - peak) / peak
        max_dd = np.min(drawdown)
        
        if metric == 'max_drawdown':
            return abs(max_dd)
        
        # For CAGR/MDD ratio, calculate CAGR if needed
        if metric == 'cagr_mdd_ratio':
            if abs(max_dd) > 0 and n_periods > 1:
                timeframe_factor = 252  # Adjust based on timeframe
                cagr = (final_equity / initial_balance) ** (timeframe_factor / n_periods) - 1
                return cagr / abs(max_dd)
            else:
                return 0
        
        # For Ulcer Index calculation
        if metric == 'ulcer_index':
            # Calculate squared drawdowns
            squared_dd = np.square(drawdown)
            # Calculate Ulcer Index (root mean square of drawdowns)
            return np.sqrt(np.mean(squared_dd))
    
    # For time in market calculation
    if metric == 'time_in_market':
        # Estimate time in market by looking at non-zero trade results and equity changes
        equity_changes = backtest_df['Equity'].diff().fillna(0)
        # Consider a bar as "in market" if there was a trade result or equity changed
        in_market = (trade_results != 0) | (equity_changes != 0)
        return np.mean(in_market) * 100  # As percentage
    
    # If "all" or an unrecognized metric is requested, calculate everything
    if metric == 'all':
        # Calculate metrics if not already done
        
        # Calculate log returns and std dev if not already done
        if log_returns is None:
            if n_periods > 1:
                log_returns = np.diff(np.log(backtest_df['Equity'].values))
                timeframe_factor = 252  # Adjust based on timeframe
                annualized_mean = np.mean(log_returns) * timeframe_factor
                annualized_std = np.std(log_returns) * np.sqrt(timeframe_factor)
                sharpe = annualized_mean / annualized_std if annualized_std > 0 else 0
            else:
                annualized_std = 0
                sharpe = 0
        
        # Calculate drawdown metrics if not already done
        equity_series = backtest_df['Equity'].values
        peak = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - peak) / peak
        max_dd = abs(np.min(drawdown))
        
        # Calculate Ulcer Index
        squared_dd = np.square(drawdown)
        ulcer_index = np.sqrt(np.mean(squared_dd))
        
        # Calculate CAGR if not already done
        if n_periods > 1:
            timeframe_factor = 252  # Adjust based on timeframe
            cagr = (final_equity / initial_balance) ** (timeframe_factor / n_periods) - 1
        else:
            cagr = 0
        
        # Calculate time in market
        equity_changes = backtest_df['Equity'].diff().fillna(0)
        in_market = (trade_results != 0) | (equity_changes != 0)
        time_in_market = np.mean(in_market) * 100  # As percentage
        
        # Calculate consecutive trades metrics if not already done
        if 'consecutive_wins' not in locals():
            trade_results_no_zero = [r for r in trade_results if r != 0]
            
            consecutive_wins = 0
            current_win_streak = 0
            consecutive_losses = 0
            current_loss_streak = 0
            
            for result in trade_results_no_zero:
                if result == 1:  # Win
                    current_win_streak += 1
                    current_loss_streak = 0
                    consecutive_wins = max(consecutive_wins, current_win_streak)
                elif result == -1:  # Loss
                    current_loss_streak += 1
                    current_win_streak = 0
                    consecutive_losses = max(consecutive_losses, current_loss_streak)
        
        # Calculate profit factor if not already done
        if 'profits' not in locals():
            pnl_series = backtest_df['Balance'].diff().fillna(0)
            profits = pnl_series[pnl_series > 0].sum()
            losses = abs(pnl_series[pnl_series < 0].sum())
            profit_factor = profits / losses if losses != 0 else float('inf')
        
        # Return all metrics as a dictionary
        return {
            'final_equity': final_equity,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'profit_factor': profit_factor,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'total_return': (final_equity / initial_balance) - 1 if initial_balance > 0 else 0,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'ulcer_index': ulcer_index,
            'std_dev': annualized_std,
            'cagr_mdd_ratio': cagr / max_dd if max_dd > 0 else float('inf'),
            'time_in_market': time_in_market
        }
    
    # Default return if metric not recognized
    return None

def bootstrapped_single_backtest(df, direction, risk_percentage, spread, 
                      datetime_col='Datetime', open_col='Open', high_col='High', 
                      low_col='Low', close_col='Close', volume_col='Volume',
                      open_signal_col='OpenSignal', close_signal_col='CloseSignal',
                      stop_loss=None, take_profit=None, pip_value=0.0001,
                        sample_size=1500, sample_number=100, objective_metric='mean_cagr'):
    # empty dataframe to store results
    results_df = pd.DataFrame()

    # create valid start index for the sample size
    max_index = len(df) - sample_size - 1
    # loop through the sample number
    for i in range(sample_number):
        # create a random start index for the sample size
        start_index = np.random.randint(0, max_index)
        end_index = start_index + sample_size
        # create a sample dataframe from the original dataframe
        sample_df = df.iloc[start_index:end_index].copy()
        
        # call the backtest strategy function on the sample dataframe
        result_df = single_backtest(df=sample_df, direction=direction, risk_percentage=risk_percentage, spread=spread,
                                      datetime_col=datetime_col, open_col=open_col, high_col=high_col,
                                      low_col=low_col, close_col=close_col, volume_col=volume_col,
                                      open_signal_col=open_signal_col, close_signal_col=close_signal_col,
                                      stop_loss=stop_loss, take_profit=take_profit, pip_value=pip_value)

        sample_metrics = calculate_trade_metrics(result_df, metric= "all")
        # add the sample metrics to the results dataframe
        results_df = pd.concat([results_df, pd.DataFrame(sample_metrics, index=[0])], ignore_index=True)
        
    # Now summarise the results dataframe
    summary_dict = {
        'mean_cagr': np.exp(np.mean(np.log(1 + results_df['cagr'].dropna()))) - 1,
        'mean_sharpe': np.exp(np.mean(np.log(1 + results_df['sharpe'].dropna()))) - 1,
        'mean_max_drawdown': results_df['max_drawdown'].mean(),
        'mean_profit_factor': results_df['profit_factor'].mean(),
        'mean_win_rate': results_df['win_rate'].mean(),
        'mean_time_in_market': results_df['time_in_market'].mean(),
        'mean_std_dev': results_df['std_dev'].mean(),
        'mean_ulcer_index': results_df['ulcer_index'].mean(),
        'mean_cagr_mdd_ratio': results_df['cagr_mdd_ratio'].mean(),
        'mean_max_consecutive_wins': results_df['max_consecutive_wins'].mean(),
        'mean_max_consecutive_losses': results_df['max_consecutive_losses'].mean(),
        'mean_total_trades': results_df['total_trades'].mean(),
        'mean_winning_trades': results_df['winning_trades'].mean(),
        'mean_losing_trades': results_df['losing_trades'].mean(),
        'mean_total_return': results_df['total_return'].mean(),
        'mean_profit_factor': results_df['profit_factor'].mean(),
    }
    
    if objective_metric in summary_dict.keys():
        return summary_dict[objective_metric]
    else:
        return summary_dict
        


#%% Updated Functions

def FX_single_backtest(df, risk_percentage=2, spread=1.0, 
                      datetime_col='Datetime', open_col='Open', high_col='High', 
                      low_col='Low', close_col='Close', volume_col='Volume',
                      open_buy_signal_col='OpenBuySignal', open_sell_signal_col='OpenSellSignal', 
                      close_buy_signal_col=None, close_sell_signal_col=None,
                      stop_loss=None, take_profit=None, 
                      stop_type='fixed', stop_unit='pips', take_profit_unit='pips',
                      trailing_step=None, trailing_bars=None,
                      bid_spread_col=None, ask_spread_col=None, slippage_col=None,
                      pip_value=0.0001):
    """
    Backtests a trading strategy with support for simultaneous long and short positions.
    
    This function uses NumPy for faster calculations and provides detailed metrics on trading performance.
    It supports both fixed and trailing stops, percentage and pip-based risk management, and
    accounts for spread and slippage in trade execution.
    
    Signals are expected in the dataframe as 1 for action and 0 for no signal.
    The signals should be in the same row as the OHLCV data they apply to. Signals generated
    from the close of the previous bar should be shifted down to align with the next bar,
    as trades open/close at the open price of the bar containing the signal.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with OHLCV data and signal columns
    risk_percentage : float or str
        If float: Percentage of balance to risk per position (0-100)
        If str: Column name in df containing risk percentage values for each bar
    spread : float
        Fixed spread in pips (used if bid_spread_col and ask_spread_col are None)
    datetime_col, open_col, high_col, low_col, close_col, volume_col : str
        Column names for OHLCV data
    open_buy_signal_col : str or None
        Column name for buy position open signals (1 for signal, 0 for no signal)
    open_sell_signal_col : str or None
        Column name for sell position open signals (1 for signal, 0 for no signal)
    close_buy_signal_col : str or None
        Column name for buy position close signals (1 for signal, 0 for no signal)
        If None, positions are only closed by SL/TP or at the end of data
    close_sell_signal_col : str or None
        Column name for sell position close signals (1 for signal, 0 for no signal)
        If None, positions are only closed by SL/TP or at the end of data
    stop_loss : float or None
        Stop loss value in pips or percentage (None to ignore stop loss)
    take_profit : float or None
        Take profit value in pips or percentage (None to ignore take profit)
    stop_type : str
        'fixed' for regular stop-loss or 'trailing' for trailing stop
    stop_unit : str
        'pips' for pip-based stop loss or 'percentage' for percentage-based stop loss
    take_profit_unit : str
        'pips' for pip-based take profit or 'percentage' for percentage-based take profit
    trailing_step : float or None
        Minimum price movement in pips/percentage required to adjust trailing stop
    trailing_bars : int or None
        Number of bars between trailing stop adjustments (None = adjust on each bar)
    bid_spread_col : str or None
        Column name for variable bid spread in pips (used for sell orders)
    ask_spread_col : str or None
        Column name for variable ask spread in pips (used for buy orders)
    slippage_col : str or None
        Column name for variable slippage in pips (added to buy price, subtracted from sell price)
    pip_value : float
        Value of one pip in price terms (default: 0.0001 for most forex pairs)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with datetime, balance, equity, and trade result columns
    """
    # Convert to numpy for faster processing
    n = len(df)
    datetime_array = df[datetime_col].values
    open_array = df[open_col].values.astype(np.float32)
    high_array = df[high_col].values.astype(np.float32)
    low_array = df[low_col].values.astype(np.float32)
    close_array = df[close_col].values.astype(np.float32)
    
    # Set up bid/ask spread arrays
    if bid_spread_col is not None and bid_spread_col in df.columns:
        bid_spread_array = df[bid_spread_col].values.astype(np.float32)
    else:
        bid_spread_array = np.full(n, spread, dtype=np.float32)
    
    if ask_spread_col is not None and ask_spread_col in df.columns:
        ask_spread_array = df[ask_spread_col].values.astype(np.float32)
    else:
        ask_spread_array = np.full(n, spread, dtype=np.float32)
    
    # Set up slippage array
    if slippage_col is not None and slippage_col in df.columns:
        slippage_array = df[slippage_col].values.astype(np.float32)
    else:
        slippage_array = np.zeros(n, dtype=np.float32)
    
    # Handle risk percentage - check if it's a string (column name) or a float
    if isinstance(risk_percentage, str):
        if risk_percentage in df.columns:
            risk_percentage_array = df[risk_percentage].values.astype(np.float32)
        else:
            raise ValueError(f"Column '{risk_percentage}' not found in dataframe")
    else:
        risk_percentage_array = np.full(n, risk_percentage, dtype=np.float32)
    
    # Validate stop and take profit units
    if stop_unit not in ['pips', 'percentage']:
        raise ValueError("stop_unit must be 'pips' or 'percentage'")
    
    if take_profit_unit not in ['pips', 'percentage']:
        raise ValueError("take_profit_unit must be 'pips' or 'percentage'")
    
    # Setup signal arrays - check for open signals separately from close signals
    has_buy_open_signals = open_buy_signal_col is not None and open_buy_signal_col in df.columns
    has_sell_open_signals = open_sell_signal_col is not None and open_sell_signal_col in df.columns
    
    # Ensure at least one direction is provided
    if not has_buy_open_signals and not has_sell_open_signals:
        raise ValueError("At least one open signal direction (buy or sell) must be provided")
    
    # Set up arrays for buy signals
    if has_buy_open_signals:
        open_buy_array = df[open_buy_signal_col].values.astype(np.int8)
    else:
        open_buy_array = np.zeros(n, dtype=np.int8)
    
    # Set up arrays for sell signals
    if has_sell_open_signals:
        open_sell_array = df[open_sell_signal_col].values.astype(np.int8)
    else:
        open_sell_array = np.zeros(n, dtype=np.int8)
    
    # Check for close signals separately
    has_buy_close_signals = close_buy_signal_col is not None and close_buy_signal_col in df.columns
    has_sell_close_signals = close_sell_signal_col is not None and close_sell_signal_col in df.columns
    
    # Set up arrays for close signals (zeros if not provided)
    if has_buy_close_signals:
        close_buy_array = df[close_buy_signal_col].values.astype(np.int8)
    else:
        close_buy_array = np.zeros(n, dtype=np.int8)
    
    if has_sell_close_signals:
        close_sell_array = df[close_sell_signal_col].values.astype(np.int8)
    else:
        close_sell_array = np.zeros(n, dtype=np.int8)
    
    # Initialize arrays to store results (using float32 for better performance)
    balance_array = np.ones(n, dtype=np.float32)
    equity_array = np.ones(n, dtype=np.float32)
    trade_result_array = np.zeros(n, dtype=np.int8)  # 0=no trade, 1=win, -1=loss
    
    # Precompute flags for performance
    stop_unit_is_percentage = (stop_unit == 'percentage')
    take_profit_unit_is_percentage = (take_profit_unit == 'percentage')
    stop_type_is_trailing = (stop_type == 'trailing')
    has_stop_loss = stop_loss is not None
    has_take_profit = take_profit is not None
    
    # Initialize trading variables
    current_balance = 1.0
    current_equity = 1.0
    
    # Buy position variables
    buy_trade_open = False
    buy_entry_price = 0.0
    buy_stop_loss_price = None
    buy_take_profit_price = None
    buy_position_size = 0.0
    buy_actual_entry_price = 0.0
    buy_highest_price = 0.0
    buy_entry_bar = 0
    buy_last_trail_bar = 0
    
    # Sell position variables
    sell_trade_open = False
    sell_entry_price = 0.0
    sell_stop_loss_price = None
    sell_take_profit_price = None
    sell_position_size = 0.0
    sell_actual_entry_price = 0.0
    sell_lowest_price = float('inf')
    sell_entry_bar = 0
    sell_last_trail_bar = 0
    
    # Main trading loop
    for i in range(n):
        # Get current values
        current_open = open_array[i]
        current_high = high_array[i]
        current_low = low_array[i]
        current_close = close_array[i]
        current_risk_percentage = risk_percentage_array[i]
        current_bid_spread = bid_spread_array[i]
        current_ask_spread = ask_spread_array[i]
        current_slippage = slippage_array[i]
        
        # Check for buy signal - don't check close_buy_array to avoid error when it's None
        if not buy_trade_open and open_buy_array[i] == 1:
            # Open a buy trade at the open price of current bar
            buy_entry_price = current_open
            buy_highest_price = buy_entry_price
            buy_entry_bar = i
            buy_last_trail_bar = i
            
            # Adjust entry price for spread and slippage (buy at ask + slippage)
            buy_actual_entry_price = buy_entry_price + (current_ask_spread * pip_value) + (current_slippage * pip_value)
            
            # Set stop loss price based on unit
            if has_stop_loss:
                if not stop_unit_is_percentage:
                    # Pip-based stop loss
                    buy_stop_loss_price = buy_entry_price - (stop_loss * pip_value)
                else:
                    # Percentage-based stop loss
                    buy_stop_loss_price = buy_entry_price * (1 - stop_loss / 100)
            
            # Set take profit price based on unit
            if has_take_profit:
                if not take_profit_unit_is_percentage:
                    # Pip-based take profit
                    buy_take_profit_price = buy_entry_price + (take_profit * pip_value)
                else:
                    # Percentage-based take profit
                    buy_take_profit_price = buy_entry_price * (1 + take_profit / 100)
            
            # Calculate position size based on risk percentage
            if has_stop_loss:
                risk_amount = current_balance * (current_risk_percentage / 100)
                
                if not stop_unit_is_percentage:
                    pip_risk = stop_loss * pip_value  # Convert pips to price
                else:
                    pip_risk = buy_entry_price * stop_loss / 100  # Convert percentage to price
                    
                buy_position_size = risk_amount / pip_risk if pip_risk > 0 else 0
            else:
                # Without stop loss, risk a fixed percentage of balance
                risk_amount = current_balance * (current_risk_percentage / 100)
                default_pip_risk = 100 * pip_value
                buy_position_size = risk_amount / default_pip_risk
            
            buy_trade_open = True
        
        # Check for sell signal - don't check close_sell_array to avoid error when it's None
        if not sell_trade_open and open_sell_array[i] == 1:
            # Open a sell trade at the open price of current bar
            sell_entry_price = current_open
            sell_lowest_price = sell_entry_price
            sell_entry_bar = i
            sell_last_trail_bar = i
            
            # Adjust entry price for spread and slippage (sell at bid - slippage)
            sell_actual_entry_price = sell_entry_price - (current_bid_spread * pip_value) - (current_slippage * pip_value)
            
            # Set stop loss price based on unit
            if has_stop_loss:
                if not stop_unit_is_percentage:
                    # Pip-based stop loss
                    sell_stop_loss_price = sell_entry_price + (stop_loss * pip_value)
                else:
                    # Percentage-based stop loss
                    sell_stop_loss_price = sell_entry_price * (1 + stop_loss / 100)
            
            # Set take profit price based on unit
            if has_take_profit:
                if not take_profit_unit_is_percentage:
                    # Pip-based take profit
                    sell_take_profit_price = sell_entry_price - (take_profit * pip_value)
                else:
                    # Percentage-based take profit
                    sell_take_profit_price = sell_entry_price * (1 - take_profit / 100)
            
            # Calculate position size based on risk percentage
            if has_stop_loss:
                risk_amount = current_balance * (current_risk_percentage / 100)
                
                if not stop_unit_is_percentage:
                    pip_risk = stop_loss * pip_value  # Convert pips to price
                else:
                    pip_risk = sell_entry_price * stop_loss / 100  # Convert percentage to price
                    
                sell_position_size = risk_amount / pip_risk if pip_risk > 0 else 0
            else:
                # Without stop loss, risk a fixed percentage of balance
                risk_amount = current_balance * (current_risk_percentage / 100)
                default_pip_risk = 100 * pip_value
                sell_position_size = risk_amount / default_pip_risk
            
            sell_trade_open = True
        
        # Process buy position if open
        buy_pnl = 0
        if buy_trade_open:
            # Update highest price for trailing stop
            if current_high > buy_highest_price:
                buy_highest_price = current_high
                
                # Adjust trailing stop if applicable
                if stop_type_is_trailing and has_stop_loss:
                    # Check if enough bars have passed since last adjustment
                    bars_condition = True
                    if trailing_bars is not None:
                        bars_condition = (i - buy_last_trail_bar) >= trailing_bars
                    
                    # Check if price has moved enough since entry
                    price_condition = True
                    if trailing_step is not None:
                        if not stop_unit_is_percentage:
                            # Pip-based trailing step
                            price_movement = buy_highest_price - buy_entry_price
                            price_condition = price_movement >= (trailing_step * pip_value)
                        else:
                            # Percentage-based trailing step
                            price_movement_pct = (buy_highest_price / buy_entry_price - 1) * 100
                            price_condition = price_movement_pct >= trailing_step
                    
                    # Only update if conditions are met
                    if bars_condition and price_condition:
                        # Update stop loss to trail the highest price
                        if not stop_unit_is_percentage:
                            new_stop = buy_highest_price - (stop_loss * pip_value)
                        else:
                            new_stop = buy_highest_price * (1 - stop_loss / 100)
                        
                        # Only move stop up, never down
                        if buy_stop_loss_price is None or new_stop > buy_stop_loss_price:
                            buy_stop_loss_price = new_stop
                            buy_last_trail_bar = i
            
            # Check for closing conditions
            close_by_signal = close_buy_array[i] == 1
            close_by_sl = has_stop_loss and current_low <= buy_stop_loss_price
            close_by_tp = has_take_profit and current_high >= buy_take_profit_price
            
            # Handle trade closure
            if close_by_sl:
                # Stop loss takes precedence
                buy_close_price = buy_stop_loss_price - (current_bid_spread * pip_value) - (current_slippage * pip_value)
                buy_pnl = (buy_close_price - buy_actual_entry_price) * buy_position_size
                buy_trade_open = False
                trade_result_array[i] = -1 if trade_result_array[i] == 0 else trade_result_array[i]
                
            elif close_by_tp:
                # Take profit hit
                buy_close_price = buy_take_profit_price - (current_bid_spread * pip_value) - (current_slippage * pip_value)
                buy_pnl = (buy_close_price - buy_actual_entry_price) * buy_position_size
                buy_trade_open = False
                trade_result_array[i] = 1 if trade_result_array[i] == 0 else trade_result_array[i]
                
            elif close_by_signal:
                # Close on signal
                buy_close_price = current_close - (current_bid_spread * pip_value) - (current_slippage * pip_value)
                buy_pnl = (buy_close_price - buy_actual_entry_price) * buy_position_size
                buy_trade_open = False
                if trade_result_array[i] == 0:
                    trade_result_array[i] = 1 if buy_pnl > 0 else -1
                
            elif i == n - 1:
                # Close at end of data
                buy_close_price = current_close - (current_bid_spread * pip_value) - (current_slippage * pip_value)
                buy_pnl = (buy_close_price - buy_actual_entry_price) * buy_position_size
                buy_trade_open = False
                if trade_result_array[i] == 0:
                    trade_result_array[i] = 1 if buy_pnl > 0 else -1
        
        # Process sell position if open
        sell_pnl = 0
        if sell_trade_open:
            # Update lowest price for trailing stop
            if current_low < sell_lowest_price:
                sell_lowest_price = current_low
                
                # Adjust trailing stop if applicable
                if stop_type_is_trailing and has_stop_loss:
                    # Check for trailing conditions
                    bars_condition = True
                    if trailing_bars is not None:
                        bars_condition = (i - sell_last_trail_bar) >= trailing_bars
                    
                    price_condition = True
                    if trailing_step is not None:
                        if not stop_unit_is_percentage:
                            # Pip-based trailing step
                            price_movement = sell_entry_price - sell_lowest_price
                            price_condition = price_movement >= (trailing_step * pip_value)
                        else:
                            # Percentage-based trailing step
                            price_movement_pct = (1 - sell_lowest_price / sell_entry_price) * 100
                            price_condition = price_movement_pct >= trailing_step
                    
                    # Only update if conditions are met
                    if bars_condition and price_condition:
                        # Update stop loss to trail the lowest price
                        if not stop_unit_is_percentage:
                            new_stop = sell_lowest_price + (stop_loss * pip_value)
                        else:
                            new_stop = sell_lowest_price * (1 + stop_loss / 100)
                        
                        # Only move stop down, never up
                        if sell_stop_loss_price is None or new_stop < sell_stop_loss_price:
                            sell_stop_loss_price = new_stop
                            sell_last_trail_bar = i
            
            # Check for closing conditions
            close_by_signal = close_sell_array[i] == 1
            close_by_sl = has_stop_loss and current_high >= sell_stop_loss_price
            close_by_tp = has_take_profit and current_low <= sell_take_profit_price
            
            # Handle trade closure
            if close_by_sl:
                # Stop loss takes precedence
                sell_close_price = sell_stop_loss_price + (current_ask_spread * pip_value) + (current_slippage * pip_value)
                sell_pnl = (sell_actual_entry_price - sell_close_price) * sell_position_size
                sell_trade_open = False
                trade_result_array[i] = -1 if trade_result_array[i] == 0 else trade_result_array[i]
                
            elif close_by_tp:
                # Take profit hit
                sell_close_price = sell_take_profit_price + (current_ask_spread * pip_value) + (current_slippage * pip_value)
                sell_pnl = (sell_actual_entry_price - sell_close_price) * sell_position_size
                sell_trade_open = False
                trade_result_array[i] = 1 if trade_result_array[i] == 0 else trade_result_array[i]
                
            elif close_by_signal:
                # Close on signal
                sell_close_price = current_close + (current_ask_spread * pip_value) + (current_slippage * pip_value)
                sell_pnl = (sell_actual_entry_price - sell_close_price) * sell_position_size
                sell_trade_open = False
                if trade_result_array[i] == 0:
                    trade_result_array[i] = 1 if sell_pnl > 0 else -1
                
            elif i == n - 1:
                # Close at end of data
                sell_close_price = current_close + (current_ask_spread * pip_value) + (current_slippage * pip_value)
                sell_pnl = (sell_actual_entry_price - sell_close_price) * sell_position_size
                sell_trade_open = False
                if trade_result_array[i] == 0:
                    trade_result_array[i] = 1 if sell_pnl > 0 else -1
        
        # Update balance with realized PnL
        current_balance += buy_pnl + sell_pnl
        
        # Calculate unrealized PnL for equity calculation
        unrealized_buy_pnl = 0
        if buy_trade_open:
            current_bid_price = current_close - (current_bid_spread * pip_value)
            unrealized_buy_pnl = (current_bid_price - buy_actual_entry_price) * buy_position_size
        
        unrealized_sell_pnl = 0
        if sell_trade_open:
            current_ask_price = current_close + (current_ask_spread * pip_value)
            unrealized_sell_pnl = (sell_actual_entry_price - current_ask_price) * sell_position_size
        
        # Current equity = balance + unrealized PnL
        current_equity = current_balance + unrealized_buy_pnl + unrealized_sell_pnl
        
        # Store current balance and equity for this row
        balance_array[i] = current_balance
        equity_array[i] = current_equity
    
    # Create and return the results dataframe
    result_df = pd.DataFrame({
        'Datetime': datetime_array,
        'Balance': balance_array,
        'Equity': equity_array,
        'TradeResult': trade_result_array
    })
    
    # Add phantom row with balance and equity of 1.0 at the beginning
    if len(result_df) > 0:
        # Create a phantom datetime just before the first entry
        if hasattr(result_df['Datetime'][0], 'to_pydatetime'):
            phantom_datetime = result_df['Datetime'][0].to_pydatetime() - pd.Timedelta(minutes=1)
        elif isinstance(result_df['Datetime'][0], pd.Timestamp):
            phantom_datetime = result_df['Datetime'][0] - pd.Timedelta(minutes=1)
        elif isinstance(result_df['Datetime'][0], dt):
            phantom_datetime = result_df['Datetime'][0] - pd.Timedelta(minutes=1)
        elif isinstance(result_df['Datetime'][0], str):
            phantom_datetime = pd.to_datetime(result_df['Datetime'][0]) - pd.Timedelta(minutes=1)
        else:
            phantom_datetime = "Initial"
            
        phantom_row = pd.DataFrame({
            'Datetime': [phantom_datetime],
            'Balance': [1.0],
            'Equity': [1.0],
            'TradeResult': [0]
        })
        
        # Concatenate the phantom row at the beginning
        result_df = pd.concat([phantom_row, result_df], ignore_index=True)
    
    return result_df

#%% testing FX_single_backtest function
train['OpenBuySignal'] = np.random.randint(0, 2, size=len(train))
train['CloseBuySignal'] = np.random.randint(0, 2, size=len(train))


output = FX_single_backtest(train, risk_percentage=2, spread=0, 
                      datetime_col='Time', open_col='Open', high_col='High', 
                      low_col='Low', close_col='Close', volume_col='Volume',
                      open_buy_signal_col='OpenBuySignal', open_sell_signal_col=None, 
                      close_buy_signal_col='CloseBuySignal', close_sell_signal_col=None,
                      stop_loss=5, take_profit=None, 
                      stop_type='trailing', stop_unit='percentage', take_profit_unit='pips',
                      trailing_step=None, trailing_bars=1,
                      bid_spread_col=None, ask_spread_col=None, slippage_col=None,
                      pip_value=0.0001)

output['Balance'].plot()















#%% optimising backtest_strategy with Mango
# Set up parameters for optimization. nb ALL parameters must be in list/range form for Mango
params = {
    'stop_loss': list(range(15, 50, 2)) + [None],  # Changed parameter names to remove _list
    'take_profit': list(range(15, 50, 2)) + [None],  # Changed parameter names to remove _list
    'direction': ['buy', 'sell'],
    'risk_percentage': np.arange(0.01, 0.2, 0.01),  # Changed to np.arange for range
    'spread': [1.2],
    'datetime_col': ['Time'],  # Must be a list
    'open_col': ['Open'],      # Must be a list
    'high_col': ['High'],      # Must be a list
    'low_col': ['Low'],        # Must be a list 
    'close_col': ['Close'],    # Must be a list
    'volume_col': ['Volume'],  # Must be a list
    'open_signal_col': ['OpenSignal'],  # FIXED: match parameter name in function
    'close_signal_col': ['CloseSignal'],
    'pip_value': [0.0001],     # Must be a list
    #'sample_size': [1500],     # Must be a list
    #'sample_number': [100],     # Must be a list
    'Open_lookback': range(2, 50),  # Must be a list
    'Close_lookback': range(2,50),  # Must be a list
    'metric': ['cagr']  # Must be a list
}


@scheduler.parallel(n_jobs=-1)
def objective_function(input_df = train, **params):
    input_df = input_df.copy()  # Create a fresh copy of the training data for each evaluation
    input_df['OpenSignal'] = input_df['Close'] > input_df['Close'].shift(params['Open_lookback']).shift(-1)  # Extract the input DataFrame from the dictionary
    input_df['CloseSignal'] = input_df['Close'] < input_df['Close'].shift(params['Close_lookback']).shift(-1)  # Example close signal
    
    backtest_df = backtest_strategy(
        df=input_df,
        direction=params['direction'],
        risk_percentage=params['risk_percentage'],
        spread=params['spread'],
        datetime_col=params['datetime_col'],
        open_col=params['open_col'],
        high_col=params['high_col'],
        low_col=params['low_col'],
        close_col=params['close_col'],
        volume_col=params['volume_col'],
        open_signal_col=params['open_signal_col'],
        close_signal_col=params['close_signal_col'],
        stop_loss=params['stop_loss'],
        take_profit=params['take_profit'],
        pip_value=params['pip_value'])
    
    metric = calculate_trade_metrics(backtest_df, metric=params['metric'])
    print(metric)
    # Only calculate the specific metric needed (CAGR in this case)
    return metric

# Create the tuner - note the order of arguments
tuner = Tuner(params, objective_function, {'initial_random':10,
 'num_iteration': 30})

# Run the optimization
results = tuner.maximize()
print('Best parameters:', results['best_params'])
print('Best objective value:', results['best_objective'])  # Not 'best_accuracy'

