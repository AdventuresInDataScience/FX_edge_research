#%% Imports
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime as dt
from scipy.optimize import differential_evolution
data_path = "C:/Users/malha/Documents/Data/FX/EURUSD_H1.csv"
#%% Read data
df = pd.read_csv(data_path, sep="\t", header=1, parse_dates=[0], index_col=0)
df = df.iloc[:, :5]
#df2 = df.copy()

# %% Main functions
def return_outcome(df, direction, spread, risk_reward, pips_risked):
    """
    Determine trade outcomes based on stop loss and take profit levels.
    Optimized version using NumPy for faster calculation.
    """
    # Create a copy to avoid modifying the original
    df_result = df.copy()
    
    # Convert spread to pips
    spread_pips = spread * 0.0001
    stop_pips = pips_risked * 0.0001
    take_pips = risk_reward * pips_risked * 0.0001
    
    # Calculate stop loss and take profit levels
    if direction == 'buy':
        df_result['stop_loss'] = df_result['Open'] + spread_pips - stop_pips
        df_result['take_profit'] = df_result['Open'] + spread_pips + take_pips
        sl_col, tp_col = 'Low', 'High'
    elif direction == 'sell':
        df_result['stop_loss'] = df_result['Open'] - spread_pips + stop_pips
        df_result['take_profit'] = df_result['Open'] - spread_pips - take_pips
        sl_col, tp_col = 'High', 'Low'
    else:
        raise ValueError("Direction must be either 'buy' or 'sell'")
    
    # Convert to numpy arrays for faster processing
    index_array = np.array(range(len(df_result)))
    dates = df_result.index.values
    stop_loss = df_result['stop_loss'].values
    take_profit = df_result['take_profit'].values
    
    # Arrays for the check columns
    sl_values = df_result[sl_col].values
    tp_values = df_result[tp_col].values
    
    # Initialize result arrays
    sl_hit_idx = np.full(len(df_result), len(df_result))
    tp_hit_idx = np.full(len(df_result), len(df_result))
    
    # Set the maximum look-forward window
    max_lookforward = min(1000, len(df_result))
    
    # Vectorized approach using numpy
    for i in range(len(df_result) - 1):
        # Define the forward window (limit to available data)
        end_idx = min(i + max_lookforward, len(df_result))
        
        if end_idx <= i + 1:  # Skip if we're at the end
            continue
            
        # Extract the slices we need
        future_sl = sl_values[i+1:end_idx]
        future_tp = tp_values[i+1:end_idx]
        future_idx = index_array[i+1:end_idx]
        
        # Find hits based on direction
        if direction == 'buy':
            sl_hits = future_idx[future_sl <= stop_loss[i]]
            tp_hits = future_idx[future_tp >= take_profit[i]]
        else:  # sell
            sl_hits = future_idx[future_sl >= stop_loss[i]]
            tp_hits = future_idx[future_tp <= take_profit[i]]
        
        # Record the first hit
        if len(sl_hits) > 0:
            sl_hit_idx[i] = sl_hits[0]
            
        if len(tp_hits) > 0:
            tp_hit_idx[i] = tp_hits[0]
    
    # Create result dataframe
    result = pd.DataFrame({
        'sl_idx': sl_hit_idx,
        'tp_idx': tp_hit_idx,
    }, index=df_result.index)
    
    # Filter valid trades (where either stop loss or take profit was hit)
    valid_mask = (result['sl_idx'] < len(df_result)) | (result['tp_idx'] < len(df_result))
    result = result[valid_mask].copy()
    
    # Map indices back to dates
    result['sl_id'] = result['sl_idx'].apply(lambda x: dates[x] if x < len(dates) else df_result.index[-1])
    result['tp_id'] = result['tp_idx'].apply(lambda x: dates[x] if x < len(dates) else df_result.index[-1])
    
    # Determine outcome (0 for loss, 1 for win)
    result['outcome'] = np.where(result['sl_idx'] <= result['tp_idx'], 0, 1)
    
    # Calculate trade length in bars
    result['length'] = np.minimum(
        result['sl_idx'] - result.index.get_indexer(result.index),
        result['tp_idx'] - result.index.get_indexer(result.index)
    )
    
    # Convert bar count to hours (since it's H1 data)
    result['length'] = result['length'].astype(float)
    
    return result['outcome'], result['length']

def get_metrics(outcomes, length, risk_per_trade, risk_reward):
    #1. calculate the win rate
    win_rate = outcomes.mean()
    #2. calculate the expectancy
    expectancy = ((1 + (risk_reward * risk_per_trade))**(win_rate)) * ((1 - risk_per_trade)**(1 - win_rate))
    #3. Average holding period - TO ADJUST THIS TO COMPENSATE FOR WKENDS
    avg_holding_period = length.mean()
    
    return win_rate, expectancy, avg_holding_period   

def bootstrap_metrics(outcomes, length, sample_size, sample_number, risk_per_trade, risk_reward):
    # If sample size is smaller than the number of outcomes, sample the outcomes
    if len(outcomes) > sample_size:
        # Create empty lists to store the metrics
        win_rate_list = []
        expectancy_list = []
        avg_holding_period_list = []
        # Loop through the number of samples
        for i in range(sample_number):
            # Create a new sample without modifying the original data
            sample_outcomes = outcomes.sample(sample_size, random_state=i)  # Using i for different seeds
            # Get the corresponding length values for the sampled outcomes
            sample_length = length.loc[sample_outcomes.index]
            # Calculate metrics - pass risk_reward here
            win_rate, expectancy, avg_holding_period = get_metrics(
                sample_outcomes, sample_length, risk_per_trade, risk_reward
            )
            win_rate_list.append(win_rate)
            expectancy_list.append(expectancy)
            avg_holding_period_list.append(avg_holding_period)
        
        # Calculate the mean, median and standard deviation of the metrics
        win_rate_mean = np.mean(win_rate_list)
        win_rate_median = np.median(win_rate_list)
        win_rate_std = np.std(win_rate_list)
        expectancy_mean = np.mean(expectancy_list)
        expectancy_median = np.median(expectancy_list)
        expectancy_std = np.std(expectancy_list)
        avg_holding_period_mean = np.mean(avg_holding_period_list)
        avg_holding_period_median = np.median(avg_holding_period_list)
        avg_holding_period_std = np.std(avg_holding_period_list)
        cagr = (expectancy_mean) ** (6000 / (1 + avg_holding_period_mean)) # Assuming 6000 is the number of hours in a year

        return win_rate_mean, win_rate_median, win_rate_std, expectancy_mean, expectancy_median, expectancy_std, avg_holding_period_mean, avg_holding_period_median, avg_holding_period_std, cagr
    
    # If sample size is larger than the number of outcomes, use the whole dataset
    else:
        win_rate, expectancy, avg_holding_period = get_metrics(
            outcomes, length, risk_per_trade, risk_reward
            )
        win_rate_mean = win_rate
        win_rate_median = win_rate
        win_rate_std = np.nan
        expectancy_mean = expectancy
        expectancy_median = expectancy
        expectancy_std = np.nan
        avg_holding_period_mean = avg_holding_period
        avg_holding_period_median = avg_holding_period
        avg_holding_period_std = np.nan
        cagr = (expectancy_mean) ** (6000 / (1 + avg_holding_period_mean))

        return win_rate_mean, win_rate_median, win_rate_std, expectancy_mean, expectancy_median, expectancy_std, avg_holding_period_mean, avg_holding_period_median, avg_holding_period_std, cagr

def build_summary_row(df, direction, spread, risk_reward, pips_risked, sample_size, sample_number, risk_per_trade):
    outcomes, length = return_outcome(df, direction, spread, risk_reward, pips_risked)
    # Pass risk_reward here
    wa, wm, ws, ea, em, es, ha, hm, hs, c = bootstrap_metrics(
        outcomes, length, sample_size, sample_number, risk_per_trade, risk_reward
    )
    return wa, wm, ws, ea, em, es, ha, hm, hs, c
    
def make_summary(df, spread, direction_list, risk_reward_list, pips_risked_list, sample_size, sample_number, risk_per_trade_list):
    # Empty lists
    direction_ = []
    risk_reward_ = []
    pips_risked_ = []
    risk_per_trade_ = []
    win_rate_mean_ = []
    win_rate_median_ = []
    win_rate_std_ = []
    expectancy_mean_ = []
    expectancy_median_ = []
    expectancy_std_ = []
    avg_holding_period_mean_ = []
    avg_holding_period_median_ = []
    avg_holding_period_std_ = []
    cagr_ = []

    #Loop through the lists and append the values to the lists
    for direction in direction_list:
        for risk_reward in risk_reward_list:
            for pips_risked in pips_risked_list:
                for risk_per_trade in risk_per_trade_list:
                    wa, wm, ws, ea, em, es, ha, hm, hs, c = build_summary_row(df, direction, spread, risk_reward, pips_risked, sample_size, sample_number, risk_per_trade)
                    direction_.append(direction)
                    risk_reward_.append(risk_reward)
                    pips_risked_.append(pips_risked)
                    risk_per_trade_.append(risk_per_trade)
                    win_rate_mean_.append(wa)
                    win_rate_median_.append(wm)
                    win_rate_std_.append(ws)
                    expectancy_mean_.append(ea)
                    expectancy_median_.append(em)
                    expectancy_std_.append(es)
                    avg_holding_period_mean_.append(ha)
                    avg_holding_period_median_.append(hm)
                    avg_holding_period_std_.append(hs)
                    cagr_.append(c)
    
    # Create a dataframe from the lists
    summary = pd.DataFrame({
        'direction': direction_,
        'risk_reward': risk_reward_,
        'pips_risked': pips_risked_,
        'risk_per_trade': risk_per_trade_,
        'win_rate_mean': win_rate_mean_,
        'win_rate_median': win_rate_median_,
        'win_rate_std': win_rate_std_,
        'expectancy_mean': expectancy_mean_,
        'expectancy_median': expectancy_median_,
        'expectancy_std': expectancy_std_,
        'avg_holding_period_mean': avg_holding_period_mean_,
        'avg_holding_period_median': avg_holding_period_median_,
        'avg_holding_period_std': avg_holding_period_std_,
        'CAGR': cagr_
    })

    return summary


#%% Grid search and return whole grid, and save
start_time = dt.now().time()
summary = make_summary(df = df,
                      spread = 1.5,
                      direction_list = ['buy', 'sell'],
                      risk_reward_list = [4, 4.5, 5, 5.5, 6],
                      pips_risked_list = (4,5,6,7,8,9,10),
                      sample_size = 1000,
                      sample_number = 200, 
                      risk_per_trade_list =[0.02, 0.03, 0.05, 0.06, 0.07])

end_time = dt.now().time()
print(f"Time taken: {end_time.hour - start_time.hour} hours, {end_time.minute - start_time.minute} minutes, {end_time.second - start_time.second} seconds")

#summary['kelly_value'] = summary['win_rate_mean'] - (1 - summary['win_rate_mean'])/summary['risk_reward']
#summary['kelly_expectancy'] = ((1 + (summary['risk_reward'] * summary['kelly_value']))**(summary['win_rate_mean'])) * ((1 - summary['kelly_value'])**(1 - summary['win_rate_mean']))
#summary['CAGR'] = summary['expectancy_mean'] ** (6000/(1 + summary['avg_holding_period_mean']).astype(int))
#summary = summary.sort_values(by='CAGR', ascending=False)
summary
summary.to_csv('summary.csv', index=False)


#%% Scipy Minimiser Optimization functions
def obj_function(params, df, direction, spread, sample_size, sample_number, objective):
    """
    Calculates a specified objective metric based on trading parameters and results.
    This function processes trading parameters through the build_summary_row function
    to generate various performance metrics, then returns the selected metric based on
    the objective parameter.
    Parameters:
        params (tuple): A tuple containing (risk_reward, pips_risked, risk_per_trade)
        df (pandas.DataFrame): DataFrame containing trading data
        direction (str): Trading direction, likely 'long' or 'short'
        spread (float): The spread value to use in calculations
        sample_size (int): Size of each trading sample
        sample_number (int): Number of samples to analyze
        objective (str): The metric to optimize, options include:
            - 'mean_expectancy': Negative of expected average return (for maximization)
            - 'median_expectancy': Negative of median expected return (for maximization)
            - 'std_expectancy': Standard deviation of expectancy
            - 'mean_win_rate': Negative of mean win rate (for maximization)
            - 'median_win_rate': Negative of median win rate (for maximization)
            - 'std_win_rate': Standard deviation of win rate
            - 'mean_avg_holding_period': Mean average holding period
            - 'median_avg_holding_period': Median average holding period
            - 'std_avg_holding_period': Standard deviation of average holding period
            - 'cagr': Negative of compound annual growth rate (for maximization)
    Returns:
        float or str: The value of the specified objective metric or "Invalid option"
                     if an invalid objective is provided. Note that some metrics are
                     negated to convert minimization problems to maximization.
    """
    # Unpack parameters    
    risk_reward, pips_risked, risk_per_trade = params
    wa, wm, ws, ea, em, es, ha, hm, hs, c = build_summary_row(
        df, direction, spread, risk_reward, pips_risked, sample_size, sample_number, risk_per_trade
    )

    options = {
        'mean_expectancy': -ea,
        'median_expectancy': -em,
        'std_expectancy': es,
        'mean_win_rate': -wa,
        'median_win_rate': -wm,
        'std_win_rate': ws,
        'mean_avg_holding_period': ha,
        'median_avg_holding_period': hm,
        'std_avg_holding_period': hs,
        'cagr': -c
    }

    return options.get(objective, "Invalid option")


def Optimize_expectancy(df, direction='buy', spread=1.2, sample_size=2000, sample_number=100,
                       risk_reward_bounds=(0.1, 5.0),
                       pips_risked_bounds=(5.0, 50.0),
                       risk_per_trade_bounds=(0.001, 0.1),
                       method='SLSQP',
                       objective='mean_expectancy'):
    """
    Optimize trading parameters to maximize expectancy with custom bounds.
    
    Parameters:
    -----------
    df : DataFrame
        Price data
    direction : str
        Trade direction ('buy' or 'sell')
    spread : float
        Spread in points
    sample_size, sample_number : int
        Bootstrap parameters
    risk_reward_bounds : tuple
        (min, max) for risk/reward ratio
    pips_risked_bounds : tuple
        (min, max) for pips risked
    risk_per_trade_bounds : tuple
        (min, max) for risk per trade
    method : str
        Optimization method ('SLSQP' or 'L-BFGS-B')
    """
    
    # Initial guess (middle of bounds)
    initial_params = [
        (risk_reward_bounds[0] + risk_reward_bounds[1]) / 2,
        (pips_risked_bounds[0] + pips_risked_bounds[1]) / 2,
        (risk_per_trade_bounds[0] + risk_per_trade_bounds[1]) / 2
    ]
    
    # Set bounds for the parameters
    bounds = [
        risk_reward_bounds,
        pips_risked_bounds,
        risk_per_trade_bounds
    ]
    
    # Run the optimization
    result = minimize(
        obj_function,
        initial_params,
        args=(df, direction, spread, sample_size, sample_number, objective),
        method=method,
        bounds=bounds
    )
    
    # Extract optimized parameters
    opt_risk_reward, opt_pips_risked, opt_risk_per_trade = result.x
    
    # Calculate the expectancy with optimized parameters
    wa, wm, ws, ea, em, es, ha, hm, hs = build_summary_row(
        df, direction, spread, opt_risk_reward, opt_pips_risked, 
        sample_size, sample_number, opt_risk_per_trade
    )
    
    return {
        'optimal_risk_reward': opt_risk_reward,
        'optimal_pips_risked': opt_pips_risked,
        'optimal_risk_per_trade': opt_risk_per_trade,
        'expectancy': ea,
        'win_rate': wa,
        'avg_holding_period': ha,
        'optimization_success': result.success,
        'message': result.message
    }
# %% Test Scipy Minimise Optimiser
best_params = Optimize_expectancy(df = df,
                                 direction = 'buy',
                                 spread = 0.6,
                                 sample_size = 1000,
                                 sample_number = 100,
                                 risk_reward_bounds=(0.1, 5.0),
                                 pips_risked_bounds=(5.0, 50.0),
                                 risk_per_trade_bounds=(0.001, 0.1),
                                 method='SLSQP',
                                 objective='mean_win_rate')
best_params

# %% Differential Evolution Optimiser Functions
def optimize_with_differential_evolution(df, direction='buy', spread=1.2, sample_size=2000, sample_number=100,
                                      risk_reward_bounds=(0.1, 5.0),
                                      pips_risked_bounds=(5.0, 50.0),
                                      risk_per_trade_bounds=(0.001, 0.1),
                                      objective='mean_expectancy',
                                      popsize=15,
                                      mutation=(0.5, 1.0),
                                      recombination=0.7,
                                      maxiter=20,
                                      tol=0.01,
                                      polish=True):
    """
    Optimize trading parameters using Differential Evolution algorithm.
    
    Parameters:
    -----------
    df : DataFrame
        Price data
    direction : str
        Trade direction ('buy' or 'sell')
    spread : float
        Spread in points
    sample_size, sample_number : int
        Bootstrap parameters
    risk_reward_bounds : tuple
        (min, max) for risk/reward ratio
    pips_risked_bounds : tuple
        (min, max) for pips risked
    risk_per_trade_bounds : tuple
        (min, max) for risk per trade
    objective : str
        Objective function to minimize (e.g., 'mean_expectancy', 'cagr')
    popsize : int
        Population size for differential evolution
    mutation : float or tuple(float, float)
        Mutation constant or bounds for adaptive mutation
    recombination : float
        Crossover probability
    maxiter : int
        Maximum number of iterations
    tol : float
        Relative tolerance for convergence
    polish : bool
        Whether to polish the result with local optimization
    
    Returns:
    --------
    dict : Dictionary containing optimized parameters and results
    """
    # Set bounds for differential evolution
    bounds = [
        risk_reward_bounds,
        pips_risked_bounds,
        risk_per_trade_bounds
    ]
    
    # Define a wrapper for our objective function to use with differential_evolution
    def objective_wrapper(params):
        return obj_function(params, df, direction, spread, sample_size, sample_number, objective)
    
    # Run differential evolution
    result = differential_evolution(
        func=objective_wrapper,
        bounds=bounds,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        maxiter=maxiter,
        tol=tol,
        polish=polish,
        disp=True  # Display progress
    )
    
    # Extract optimized parameters
    opt_risk_reward, opt_pips_risked, opt_risk_per_trade = result.x
    
    # Calculate metrics with optimized parameters
    wa, wm, ws, ea, em, es, ha, hm, hs, c = build_summary_row(
        df, direction, spread, opt_risk_reward, opt_pips_risked, 
        sample_size, sample_number, opt_risk_per_trade
    )
    
    # Return dictionary of results
    return {
        'optimal_risk_reward': opt_risk_reward,
        'optimal_pips_risked': opt_pips_risked, 
        'optimal_risk_per_trade': opt_risk_per_trade,
        'expectancy_mean': ea,
        'expectancy_median': em,
        'win_rate_mean': wa,
        'win_rate_median': wm,
        'avg_holding_period_mean': ha,
        'cagr': c,
        'optimization_success': result.success,
        'function_evals': result.nfev,
        'iterations': result.nit,
        'objective_value': result.fun
    }

#%% Test Differential Evolution Optimiser
start_time = dt.now().time()
de_results_buy = optimize_with_differential_evolution(
    df=df,
    direction='buy',
    spread=01.2,
    sample_size=1000,
    sample_number=100,
    risk_reward_bounds=(0.5, 5.0),
    pips_risked_bounds=(5.0, 25.0),
    risk_per_trade_bounds=(0.01, 0.1),
    objective='cagr',
    popsize=15,
    maxiter=50  # Increase for potentially better results
)
print(de_results_buy)
end_time = dt.now().time()
print(f"Time taken: {end_time.hour - start_time.hour} hours, {end_time.minute - start_time.minute} minutes, {end_time.second - start_time.second} seconds")



start_time = dt.now().time()
de_results_sell = optimize_with_differential_evolution(
    df=df,
    direction='buy',
    spread=1.2,
    sample_size=1000,
    sample_number=100,
    risk_reward_bounds=(0.5, 5.0),
    pips_risked_bounds=(8.0, 25.0),
    risk_per_trade_bounds=(0.01, 0.1),
    objective='cagr',
    popsize=15,
    maxiter=50  # Increase for potentially better results
)
print(de_results_sell)
end_time = dt.now().time()
print(f"Time taken: {end_time.hour - start_time.hour} hours, {end_time.minute - start_time.minute} minutes, {end_time.second - start_time.second} seconds")

# Save the results to a CSV file
de_results_buy_df = pd.DataFrame(de_results_buy, index=[0])
de_results_sell_df = pd.DataFrame(de_results_sell, index=[0])
de_results = pd.concat([de_results_buy_df, de_results_sell_df], axis=0)
de_results.to_csv('de_results.csv', index=False)
# %% Backtest function
# This function backtests a simple trading strategy with stop loss and take profit.
# It uses numpy for faster calculations and can handle both buy and sell trades.
# The function takes a dataframe with OHLCV data and calculates the balance and equity over time.
# It also allows for customization of the stop loss, take profit, and risk percentage.
# The function returns a dataframe with the datetime, balance, and equity columns.

def backtest_strategy(df, stop_loss, take_profit, direction, risk_percentage, spread, 
                      datetime_col='Datetime', open_col='Open', high_col='High', 
                      low_col='Low', close_col='Close', volume_col='Volume', 
                      pip_value=0.0001):
    """
    Backtests a simple trading strategy with stop loss and take profit.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with OHLCV data
    stop_loss : float
        Stop loss in pips
    take_profit : float
        Take profit in pips
    direction : str
        'buy' or 'sell'
    risk_percentage : float
        Percentage of balance to risk (0-100)
    spread : float
        Spread in pips
    datetime_col, open_col, high_col, low_col, close_col, volume_col : str
        Column names for OHLCV data
    pip_value : float
        Value of one pip in price terms (default: 0.0001 for most forex pairs)
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe with datetime, balance, and equity columns
    """
    # Convert to numpy for faster processing
    datetime_array = df[datetime_col].values
    open_array = df[open_col].values.astype(np.float64)
    high_array = df[high_col].values.astype(np.float64)
    low_array = df[low_col].values.astype(np.float64)
    close_array = df[close_col].values.astype(np.float64)
    
    # Initialize arrays to store results
    n = len(df)
    balance_array = np.ones(n)  # Starting balance of 1
    equity_array = np.ones(n)   # Starting equity of 1
    
    # Initialize trading variables
    current_balance = 1.0
    current_equity = 1.0
    trade_open = False
    entry_price = 0.0
    stop_loss_price = 0.0
    take_profit_price = 0.0
    position_size = 0.0
    actual_entry_price = 0.0
    
    # Loop through each row
    for i in range(n):
        # If no trade is open, open one
        if not trade_open:
            # Open a trade
            entry_price = open_array[i]
            
            # Set stop loss and take profit prices
            if direction.lower() == 'buy':
                # For buy: entry price - stop loss pips
                stop_loss_price = entry_price - (stop_loss * pip_value)
                # For buy: entry price + take profit pips
                take_profit_price = entry_price + (take_profit * pip_value)
                # Adjust entry price for spread (buy at ask, which is higher)
                actual_entry_price = entry_price + (spread * pip_value)
            else:  # sell
                # For sell: entry price + stop loss pips
                stop_loss_price = entry_price + (stop_loss * pip_value)
                # For sell: entry price - take profit pips
                take_profit_price = entry_price - (take_profit * pip_value)
                # Adjust entry price for spread (sell at bid, which is lower)
                actual_entry_price = entry_price - (spread * pip_value)
            
            # Calculate position size based on risk percentage
            risk_amount = current_balance * (risk_percentage / 100)
            pip_risk = stop_loss * pip_value  # Convert pips to price
            position_size = risk_amount / pip_risk if pip_risk > 0 else 0
            
            trade_open = True
        
        # If a trade is open, check if it has hit stop loss or take profit
        if trade_open:
            current_close = close_array[i]
            current_high = high_array[i]
            current_low = low_array[i]
            
            # Check if both stop loss and take profit are hit in the same candle
            sl_hit = False
            tp_hit = False
            
            if direction.lower() == 'buy':
                sl_hit = current_low <= stop_loss_price
                tp_hit = current_high >= take_profit_price
            else:  # sell
                sl_hit = current_high >= stop_loss_price
                tp_hit = current_low <= take_profit_price
            
            # If both are hit, assume stop loss was hit first (as per requirement)
            if sl_hit and tp_hit:
                # Close the trade at stop loss
                if direction.lower() == 'buy':
                    pnl = (stop_loss_price - actual_entry_price) * position_size
                else:  # sell
                    pnl = (actual_entry_price - stop_loss_price) * position_size
                
                current_balance += pnl
                current_equity = current_balance
                trade_open = False
            
            # If only stop loss is hit
            elif sl_hit:
                # Close the trade at stop loss
                if direction.lower() == 'buy':
                    pnl = (stop_loss_price - actual_entry_price) * position_size
                else:  # sell
                    pnl = (actual_entry_price - stop_loss_price) * position_size
                
                current_balance += pnl
                current_equity = current_balance
                trade_open = False
            
            # If only take profit is hit
            elif tp_hit:
                # Close the trade at take profit
                if direction.lower() == 'buy':
                    pnl = (take_profit_price - actual_entry_price) * position_size
                else:  # sell
                    pnl = (actual_entry_price - take_profit_price) * position_size
                
                current_balance += pnl
                current_equity = current_balance
                trade_open = False
            
            # If neither stop loss nor take profit is hit, update equity
            else:
                # Calculate unrealized PnL
                if direction.lower() == 'buy':
                    unrealized_pnl = (current_close - actual_entry_price) * position_size
                else:  # sell
                    unrealized_pnl = (actual_entry_price - current_close) * position_size
                
                current_equity = current_balance + unrealized_pnl
                
                # If this is the last candle, close the trade at current close
                if i == n - 1:
                    current_balance += unrealized_pnl
                    current_equity = current_balance
                    trade_open = False
        
        # Store current balance and equity
        balance_array[i] = current_balance
        equity_array[i] = current_equity
    
    # Create and return the results dataframe
    result_df = pd.DataFrame({
        'Datetime': datetime_array,
        'Balance': balance_array,
        'Equity': equity_array
    })
    
    # Add phantom row with balance and equity of 1.0 at the beginning
    # Create a phantom datetime just before the first entry
    if len(result_df) > 0:
        # If there's datetime data, use a time just before the first entry
        if hasattr(result_df['Datetime'][0], 'to_pydatetime'):
            phantom_datetime = result_df['Datetime'][0].to_pydatetime() - pd.Timedelta(minutes=1)
        elif isinstance(result_df['Datetime'][0], pd.Timestamp):
            phantom_datetime = result_df['Datetime'][0] - pd.Timedelta(minutes=1)
        elif isinstance(result_df['Datetime'][0], dt):
            phantom_datetime = result_df['Datetime'][0] - pd.Timedelta(minutes=1)
        elif isinstance(result_df['Datetime'][0], str):
            # If it's a string, convert to datetime and subtract 1 minute
            phantom_datetime = pd.to_datetime(result_df['Datetime'][0]) - pd.Timedelta(minutes=1)
        else:
            # If it's not a timestamp, just use a placeholder value
            phantom_datetime = "Initial"
            
        phantom_row = pd.DataFrame({
            'Datetime': [phantom_datetime],
            'Balance': [1.0],
            'Equity': [1.0]
        })
        
        # Concatenate the phantom row at the beginning
        result_df = pd.concat([phantom_row, result_df], ignore_index=True)
    
    return result_df
# %% Run Backtest function
# Training data, at 75% of the data
train_size = int(len(df) * 0.75)
train = df.iloc[:train_size].copy().reset_index()
# Test data, at 25% of the data
test = df.iloc[train_size:].copy().reset_index()



output = backtest_strategy(df = train, stop_loss = 6, take_profit = 36, 
                           direction = 'buy', risk_percentage = 0.02, spread = 1.5,
                      datetime_col='Time', open_col='Open', high_col='High', 
                      low_col='Low', close_col='Close', volume_col='Volume', 
                      pip_value=0.0001)


#%% Grid Search Direct Backtests Function
def grid_search_backtests(df=df,
                          stop_loss_list=[6],
                          take_profit_list=[36],
                          direction_list=['buy', 'sell'],
                          risk_percentage_list=[0.02],
                          spread_list=[1.5],
                          datetime_col='Time', open_col='Open', high_col='High',
                          low_col='Low', close_col='Close', volume_col='Volume',
                          pip_value=0.0001,
                          sample_size=1500,
                          sample_number=100):
    '''
    Perform a grid search over multiple parameters for backtesting.
    This function will iterate over the provided parameter lists and perform backtests
    using the backtest_strategy function. It will collect the results and return a DataFrame.
      
    '''
    # Use different names for output collection lists
    cagr_values = []
    log_std_values = []
    std_cagr_values = []
    sharpe_values = []
    std_sharpe_values = []
    
    # Use different names for the final output lists
    result_stop_loss = []
    result_take_profit = []
    result_direction = []
    result_risk_percentage = []
    result_spread = []
    result_cagr = []
    result_log_std = []
    result_std_cagr = []
    result_sharpe = []
    result_std_sharpe = []

    for stop_loss in stop_loss_list:
        for take_profit in take_profit_list:
            for direction in direction_list:
                for risk_percentage in risk_percentage_list:
                    for spread in spread_list:
                        # Reset collection lists for each parameter combination
                        cagr_values = []
                        log_std_values = []
                        std_cagr_values = []
                        sharpe_values = []
                        std_sharpe_values = []
                        
                        max_index = len(df) - sample_size - 1
                        for i in range(sample_number):
                            start_index = np.random.randint(0, max_index)
                            end_index = start_index + sample_size
                            sample_df = df.iloc[start_index:end_index].copy()
                            
                            output = backtest_strategy(df=sample_df, 
                                                      stop_loss=stop_loss, 
                                                      take_profit=take_profit,
                                                      direction=direction, 
                                                      risk_percentage=risk_percentage, 
                                                      spread=spread,
                                                      datetime_col=datetime_col, 
                                                      open_col=open_col, 
                                                      high_col=high_col,
                                                      low_col=low_col, 
                                                      close_col=close_col, 
                                                      volume_col=volume_col,
                                                      pip_value=pip_value)
                            
                            # Calculate metrics
                            cagr = (output['Equity'].iloc[-1]) ** (6000 / len(output)) - 1
                            log_std = np.std(np.log(output['Balance'].iloc[1:]) * (6000 / len(output)))
                            sharpe = cagr / log_std if log_std != 0 else 0
                            
                            cagr_values.append(cagr)
                            log_std_values.append(log_std)
                            sharpe_values.append(sharpe)
                        
                        # Calculate statistics and append to results
                        result_stop_loss.append(stop_loss)
                        result_take_profit.append(take_profit)
                        result_direction.append(direction)
                        result_risk_percentage.append(risk_percentage)
                        result_spread.append(spread)
                        result_cagr.append(np.mean(cagr_values))
                        result_log_std.append(np.mean(log_std_values))
                        result_std_cagr.append(np.std(cagr_values))
                        result_sharpe.append(np.mean(sharpe_values))
                        result_std_sharpe.append(np.std(sharpe_values))

    results_df = pd.DataFrame({
        'stop_loss': result_stop_loss,
        'take_profit': result_take_profit,
        'direction': result_direction,
        'risk_percentage': result_risk_percentage,
        'spread': result_spread,
        'mean_cagr': result_cagr,
        'mean_log_std': result_log_std,
        'std_cagr': result_std_cagr,
        'mean_sharpe': result_sharpe,
        'std_sharpe': result_std_sharpe
    })

    return results_df
# %% Run Grid Search Backtest
start_time = dt.now().time()
train_size = int(len(df) * 0.75)
train = df.iloc[:train_size].copy().reset_index()

results = grid_search_backtests(df = train,
                                 stop_loss_list = range(4,50,2),
                                 take_profit_list = range(4,50,2),
                                 direction_list = ['buy', 'sell'],
                                 risk_percentage_list = [0.01, 0.02],
                                 spread_list = [0.6, 1.2, 1.5],
                                 datetime_col='Time', open_col='Open', high_col='High',
                                 low_col='Low', close_col='Close', volume_col='Volume',
                                 pip_value=0.0001,
                                 sample_size = 1500,
                                 sample_number = 100)

end_time = dt.now().time()
print(f"Time taken: {end_time.hour - start_time.hour} hours, {end_time.minute - start_time.minute} minutes, {end_time.second - start_time.second} seconds")

results.sort_values(by = 'cagr', ascending = False)
# %%
from mango import scheduler, Tuner

# Set up parameters for optimization
# ALL parameters must be in list/range form for Mango
params = {
    'stop_loss': range(4,101, 1),  # Changed parameter names to remove _list
    'take_profit': range(4,101, 1),  # Changed parameter names to remove _list
    'direction': ['buy', 'sell'],
    'risk_percentage': [0.01, 0.02, 0.03, 0.04, 0.05],
    'spread': [0.6, 1.2, 1.5],
    'datetime_col': ['Time'],  # Must be a list
    'open_col': ['Open'],      # Must be a list
    'high_col': ['High'],      # Must be a list
    'low_col': ['Low'],        # Must be a list 
    'close_col': ['Close'],    # Must be a list
    'volume_col': ['Volume'],  # Must be a list
    'pip_value': [0.0001],     # Must be a list
    'sample_size': [1500],     # Must be a list
    'sample_number': [100]     # Must be a list
}

@scheduler.parallel(n_jobs=-1)
def objective_function(**params):
    """
    Objective function for grid search backtests.
    This function will be called by the Tuner to evaluate the performance of the strategy.
    """
    # Create lists from individual values for grid_search_backtests
    stop_loss_list = [params['stop_loss']]
    take_profit_list = [params['take_profit']]
    direction_list = [params['direction']]
    risk_percentage_list = [params['risk_percentage']]
    spread_list = [params['spread']]
    
    # Call the grid_search_backtests function with the provided parameters
    results_df = grid_search_backtests(
        df=train,
        stop_loss_list=stop_loss_list,
        take_profit_list=take_profit_list,
        direction_list=direction_list,
        risk_percentage_list=risk_percentage_list,
        spread_list=spread_list,
        datetime_col=params['datetime_col'],
        open_col=params['open_col'],
        high_col=params['high_col'],
        low_col=params['low_col'],
        close_col=params['close_col'],
        volume_col=params['volume_col'],
        pip_value=params['pip_value'],
        sample_size=params['sample_size'],
        sample_number=params['sample_number']
    )
    
    # Return the mean CAGR as the objective metric to maximize
    if len(results_df) > 0:
        return results_df['mean_cagr'].mean()
    else:
        return -float('inf')  # Return a very bad score if no results

# Create the tuner - note the order of arguments
tuner = Tuner(params, objective_function, {'initial_random': 5,
 'num_iteration': 30})

# Run the optimization
results = tuner.maximize()
print('Best parameters:', results['best_params'])
print('Best objective value:', results['best_objective'])  # Not 'best_accuracy'
# %%
