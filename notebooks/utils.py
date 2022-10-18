import os
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels import regression
import matplotlib.pyplot as plt
from statsmodels import regression

def get_top20_stocks(path, limit=20):
    """Return top20 stocks with total traded volume"""
    stock_list = {}
    files = os.listdir(path)
    for f in files:
        stock_list[f[:-4]] = pd.read_csv(f'{path}/{f}')['volume'].sum()
    return sorted(stock_list.items())[:limit]
    
def get_stocknames(path):
    symbs = []
    files = os.listdir(path)
    for f in files:
        symbs.append(f[:-4])
    return symbs

def get_prices(stockname, path):
    df = pd.read_csv(f'{path}/{stockname}.csv', parse_dates=True)
    df.set_index(df.timestamp, inplace=True)
    df.drop(columns = ['timestamp'], inplace=True)
    return df

def fig_plotter(index, values, xlabel, ylabel, stockname):
    plt.plot(index, values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(stockname)
    # plt.show()
    
def get_returns(dataframe, field):
    if field=='all':
        return dataframe.pct_change()[1:]
    else:
        return dataframe[field].pct_change()[1:]
    
def mode(l):
    # Count the number of times each element appears in the list
    counts = {}
    for e in l:
        if e in counts:
            counts[e] += 1
        else:
            counts[e] = 1
            
    # Return the elements that appear the most times
    maxcount = 0
    modes = {}
    for (key, value) in counts.items():
        if value > maxcount:
            maxcount = value
            modes = {key}
        elif value == maxcount:
            modes.add(key)
            
    if maxcount > 1 or len(l) == 1:
        return list(modes)
    return 'No mode'

def sharpe_ratio(asset, riskfree=0):
    return np.mean(asset - riskfree)/np.std(asset - riskfree)

def linreg(X,Y):
    # Running the linear regression
    X = sm.add_constant(X)
    model = regression.linear_model.OLS(Y, X).fit()
    a = model.params[0]
    b = model.params[1]
    X = X[:, 1]

    # Return summary of the regression and plot results
    X2 = np.linspace(X.min(), X.max(), 100)
    Y_hat = X2 * b + a
    plt.scatter(X, Y, alpha=0.3) # Plot the raw data
    plt.plot(X2, Y_hat, 'r', alpha=0.9);  # Add the regression line, colored in red
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    return model.summary()