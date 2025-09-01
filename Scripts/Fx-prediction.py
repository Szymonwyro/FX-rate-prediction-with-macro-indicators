# Objective: Predicting Daily FX rates using macroeconomic indicators for selected currency pairs
# This project aims to leverage machine learning techniques to forecast foreign exchange rates by analyzing key economic indicators (CPI, Interest Rate, Unemployment Rate, GDP)
# FX pairs: EUR/USD, USD/JPY, GBP/USD

import pandas as pd
from pandas_datareader import data as pdr
import investpy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fx_pairs = ['EURUSD', 'USDJPY', 'GBPUSD']
macro_vars = {
    'CPI': 'CPIAUCNS',              # Consumer Price Index
    'Interest Rate': 'FEDFUNDS',    # Fed Funds Rate
    'Unemployment Rate': 'UNRATE',   # Unemployment Rate
    'GDP': 'GDP'                     # GDP Quarterly
}

fx_data = {}

for pair in fx_pairs:
    fx_data[pair] = pd.read_csv(f'Data/{pair}.csv', parse_dates=['Date'], index_col='Date')['Price'] #CSV files here start in 2025 so needs changing

fx_df = pd.DataFrame(fx_data)
print("FX data head:", fx_df.head())

returns_df = fx_df.pct_change().shift(-1)
returns_df = returns_df.dropna()
print("Returns data head:", returns_df.head())

macro_df = pd.DataFrame()
for name, code in macro_vars.items():
    series = pdr.DataReader(code, 'fred', start="2015-01-01", end="2025-01-01") # Some missing data here in 2025 as FX data runs after 2025-01-01
    macro_df[name] = series

macro_df = macro_df.resample('D').ffill()
macro_df = macro_df.reindex(fx_df.index).ffill()
print(macro_df.head())

data = pd.concat([returns_df, macro_df], axis=1).dropna()
print("Combined data head:", data.head())

plt.figure(figsize=(12,6))
for pair in fx_pairs:
    plt.plot(returns_df[pair], label=f'{pair} Next-Day Return')
plt.title('Next-Day FX Returns')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
for pair in fx_pairs:
    sns.kdeplot(returns_df[pair], label=f'{pair} Returns', fill=True)
plt.title('Distribution of Next-Day FX Returns')
plt.xlabel('Return')
plt.ylabel('Density')
plt.legend()
plt.show()

corr = data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation between FX Returns and Macro Variables')
plt.show()
