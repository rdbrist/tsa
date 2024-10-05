import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
