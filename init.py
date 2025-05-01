#%% Imports
import pandas as pd
import numpy as np
from scipy.optimize import minimize

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
        df_result['stop_loss'] = df_result['Open'] - spread_pips - stop_pips
        df_result['take_profit'] = df_result['Open'] + spread_pips + take_pips
        sl_col, tp_col = 'Low', 'High'
    elif direction == 'sell':
        df_result['stop_loss'] = df_result['Open'] + spread_pips + stop_pips
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
    result['outcome'] = np.where(result['sl_idx'] < result['tp_idx'], 0, 1)
    
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

        return win_rate_mean, win_rate_median, win_rate_std, expectancy_mean, expectancy_median, expectancy_std, avg_holding_period_mean, avg_holding_period_median, avg_holding_period_std
    
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

        return win_rate_mean, win_rate_median, win_rate_std, expectancy_mean, expectancy_median, expectancy_std, avg_holding_period_mean, avg_holding_period_median, avg_holding_period_std

def build_summary_row(df, direction, spread, risk_reward, pips_risked, sample_size, sample_number, risk_per_trade):
    outcomes, length = return_outcome(df, direction, spread, risk_reward, pips_risked)
    # Pass risk_reward here
    wa, wm, ws, ea, em, es, ha, hm, hs = bootstrap_metrics(
        outcomes, length, sample_size, sample_number, risk_per_trade, risk_reward
    )
    return wa, wm, ws, ea, em, es, ha, hm, hs
    
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

    #Loop through the lists and append the values to the lists
    for direction in direction_list:
        for risk_reward in risk_reward_list:
            for pips_risked in pips_risked_list:
                for risk_per_trade in risk_per_trade_list:
                    wa, wm, ws, ea, em, es, ha, hm, hs = build_summary_row(df, direction, spread, risk_reward, pips_risked, sample_size, sample_number, risk_per_trade)
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
        'avg_holding_period_std': avg_holding_period_std_
    })

    return summary

#%% Tests

#%% Grid search and return whole grid, and save
summary = make_summary(df = df,
                      spread = 1.2,
                      direction_list = ['buy', 'sell'],
                      risk_reward_list = [0.5, 1, 2],  # Or use np.array
                      pips_risked_list = (20,25,30,35,40),
                      sample_size = 2000,
                      sample_number = 200, 
                      risk_per_trade_list =[0.02, 0.05])

summary
summary.to_csv('summary.csv', index=False)


#%% Optimization functions
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
    Returns:
        float or str: The value of the specified objective metric or "Invalid option"
                     if an invalid objective is provided. Note that some metrics are
                     negated to convert minimization problems to maximization.
    """
    # Unpack parameters    
    risk_reward, pips_risked, risk_per_trade = params
    wa, wm, ws, ea, em, es, ha, hm, hs = build_summary_row(
        df, direction, spread, risk_reward, pips_risked, sample_size, sample_number, risk_per_trade
    )

    options = {
        'mean_expectancy': ea,
        'median_expectancy': em,
        'std_expectancy': es,
        'mean_win_rate': wa,
        'median_win_rate': wm,
        'std_win_rate': ws,
        'mean_avg_holding_period': ha,
        'median_avg_holding_period': hm,
        'std_avg_holding_period': hs
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
# %% Test Optimiser
best_params = Optimize_expectancy(df = df,
                                 direction = 'buy',
                                 spread = 0.6,
                                 sample_size = 2000,
                                 sample_number = 100,
                                 risk_reward_bounds=(0.1, 2.0),
                                 pips_risked_bounds=(5.0, 50.0),
                                 risk_per_trade_bounds=(0.001, 0.1),
                                 method='SLSQP',
                                 objective='mean_win_rate')
best_params
# %%
