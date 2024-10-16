import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

def split_ts(y: pd.DataFrame, ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_date = y.index.min()
    max_date = y.index.max()
    num_vals = len(y)
    split = round(num_vals * ratio)
    print(f'Min date: {min_date}')
    print(f'Max date: {max_date}')
    print(f'Number of values: {num_vals}')
    print(f'Train set size: {split}')
    print(f'Test set size: {num_vals - split}')
    ytrain = y.iloc[:split]
    ytest = y.iloc[split:]
    return ytrain, ytest

def ts_plot(y: pd.DataFrame):
    y.plot(figsize=(18,8))
    plt.show()

def ts_dist_plot(x: pd.Series):
    fg,ax = plt.subplots(nrows=2,figsize=(15,8))
    sns.boxplot(x=x, ax=ax[0])
    sns.histplot(x=x, ax=ax[1], kde=True)
    plt.show()

def p_q_result(pmax: int, qmax: int, pstep: int, qstep: int, ytrain: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p_params = range(0,pmax,pstep)
    q_params = range(0,qmax,qstep)
    mae_grid = dict()
    aicbic_df = pd.DataFrame(columns=['Order','AIC','BIC'])
    init_time = time.time()
    for p in p_params:
        mae_grid[p] = list()
        for q in q_params:
            order = (p, 0, q)
            start_time = time.time()
            model = ARIMA(ytrain, order=order).fit()
            aicbic_df = pd.concat([aicbic_df, [order, model.aic, model.bic]])
            elapsed_time = round(time.time() - start_time, 2)
            print(f"Trained ARIMA {order} in {elapsed_time} seconds.")
            y_pred = model.predict()
            mae = mean_absolute_error(ytrain, y_pred)
            mae_grid[p].append(mae)
    print(f'All permutations completed in {round(time.time() - init_time, 2)} seconds.')
    del mae_grid[0] # given you wouldn't consider an ARMA model without autoregression, also makes the heatmap more meaningful by reducing significantly different MAE values
    mae_df = pd.DataFrame(mae_grid)
    print(mae_df.round(4))
    sns.heatmap(mae_df, cmap = "Blues")
    plt.xlabel("p values")
    plt.ylabel("q values")
    plt.title("ARIMA Model Performance")
    return mae_df, aicbic_df

def run_adf(df):
    adftest = adfuller(df, autolag='AIC', regression='ct')
    print("ADF Test Results")
    print("Null Hypothesis: The series has a unit root (non-stationary)")
    print("ADF-Statistic:", adftest[0])
    print("P-Value:", adftest[1])
    print("Number of lags:", adftest[2])
    print("Number of observations:", adftest[3])
    print("Critical Values:", adftest[4])
    print("Note: If P-Value is smaller than 0.05, we reject the null hypothesis and the series is stationary")