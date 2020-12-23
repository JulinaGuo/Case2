# Case2: Price prediction and Funds set based on scenario analysis

### Overview
The purpose of the project is to predict the future trend of different funds. Based on scenario analysis and client's requirement, we can later design customized portfolio set.

### Data cleaning
```
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd

sample = pd.read_excel('/content/PORTFOLIO_PRICE_SAMPLE.xlsx', header=0)
sample.set_index('Date', inplace=True)
sample = sample.dropna()
pct_change = sample.pct_change().dropna()
```

### Client's debt:shock 70:30 preferrence
```
sample_equity = pd.DataFrame()
sample_equity['JPM_America_Equity'] = sample['JPM_America_Equity']
sample_equity['JPM_Asia_Pacific_Equity'] = sample['JPM_Asia_Pacific_Equity']
sample_equity['JPM_China_AShare_Opportunity'] = sample['JPM_China_AShare_Opportunity']
sample_equity['JPM_Emerging_Markets_Equity'] = sample['JPM_Emerging_Markets_Equity']
sample_equity['JPM_Europe_Dynamic'] = sample['JPM_Europe_Dynamic']
sample_equity['JPM_Global_Healthcare'] = sample['JPM_Global_Healthcare']
sample_equity['JPM_Global_Socially_Responsible'] = sample['JPM_Global_Socially_Responsible']
sample_equity['JPM_Japan_Equity'] = sample['JPM_Japan_Equity']
sample_equity['JPM_US_Technology'] = sample['JPM_US_Technology']
sample_equity['JPM_USD_Liquidity_VNAV'] = sample['JPM_USD_Liquidity_VNAV']

sample_debt = pd.DataFrame()
sample_debt['JPM_Emerging_Markets_Debt'] = sample['JPM_Emerging_Markets_Debt']
sample_debt['JPM_Global_Corporate_Bond'] = sample['JPM_Global_Corporate_Bond']
sample_debt['JPM_Global_Government_Bond'] = sample['JPM_Global_Government_Bond']
sample_debt['JPM_Global_High_Yield_Bond'] = sample['JPM_Global_High_Yield_Bond']

pct_change_equity = sample_equity.pct_change()
pct_change_debt = sample_debt.pct_change()

# 客戶原先的70%債30%股的資產配置表現

def equity7_debt3_weighting(pct_change):
  equity_weight = np.random.random(10)
  debt_weight = np.random.random(4)
  equity_weight = (equity_weight / np.sum(equity_weight)) * 0.3
  debt_weight = (debt_weight / np.sum(debt_weight)) * 0.7
  weight = np.append(equity_weight, debt_weight)

  # 2015-01-01 to 2020-09-17 有價格資料日期1235天
  annualized_return_rate_equity = ((1 + pct_change_equity).cumprod() ** (1 / 1229)-1)[-1:]
  annualized_return_rate_equity = annualized_return_rate_equity.iloc[0, :]
  annualized_return_rate_debt = ((1 + pct_change_debt).cumprod() ** (1 / 1229)-1)[-1:]
  annualized_return_rate_debt = annualized_return_rate_debt.iloc[0, :]
  annualized_return_rate = np.append(annualized_return_rate_equity, annualized_return_rate_debt)
  portfolio_return = weight.T.dot(annualized_return_rate) * 252  
  
  stock_cov = pct_change.cov()
  portfolio_volatility = np.sqrt(weight.T.dot(stock_cov.dot(weight)) * 252)
  return weight, portfolio_return, portfolio_volatility

n = 100000
weight, portfolio_return, portfolio_volatility = np.column_stack([equity7_debt3_weighting(pct_change) for _ in range(n)])
analysis = pd.DataFrame()
analysis['portfolio_return'] = portfolio_return
analysis['portfolio_volatility'] = portfolio_volatility

plt.figure(figsize = (12,8))
plt.scatter(portfolio_volatility , portfolio_return, c = (portfolio_return - 0.0046)/(portfolio_volatility**0.5), marker = 'o')
plt.colorbar(label = 'Sharpe ratio')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Return Rate')
plt.title('Mean-Variance Analysis of Portfolios')
plt.grid()
```

### Recommended portfolio is debt:stock 20:80
```
# 新的資產配置比例 (based on historical data)
import cvxopt as opt
from cvxopt import blas, solvers
import scipy as sci
from scipy import stats

def opt_portfolio(pct_change):
    
    n = 14
    solvers.options['show_progress'] = False
     
    N = 10000
    risk_levels = [10 ** (2 * t / N - 2) for t in range(N)]
    
    p = (((1 + pct_change).cumprod() ** (1 / 1229)) - 1)[-1:]
    p = opt.matrix(p.values * 252).T
    S = opt.matrix(pct_change.cov().values * 252)
    
    G = opt.matrix(-np.eye(n))
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    weights = [solvers.qp(2 * S, -level * p, G, h, A, b)['x'] for level in risk_levels]
    returns = np.array([blas.dot(p, x) for x in weights])
    vols = np.array([np.sqrt(blas.dot(x, S * x)) for x in weights])
    
    idx = returns / vols
    
    return idx, weights, returns, vols

opt_risks = []
opt_returns = []
opt_idx, opt_weight, opt_returns, opt_risks = opt_portfolio(pct_change)

plt.figure(figsize = (12, 6))
plt.plot(opt_risks, opt_returns, 'y-o')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Return Rate')
plt.title('Mean-Variance Analysis of Portfolios')
plt.grid()
```
![](https://i.imgur.com/0jJUuCs.png)

### Single fund prediction
```
# JPM_America_Equity 

r = 0.05 - 0.01219
v = 0.011721059240462567 + 0.01 # 未加上scenario

T = 1
N = 180 # 預測天數
dt = T / N

future = []
plt.figure(figsize = (12, 6))
for j in range(10000):
    future_price = []
    p = np.zeros((N + 1, 1))
    p[0] = 31.75
    for i in range(N):
        p[i + 1] = p[i] * np.exp((r - 0.5 * v ** 2) * dt + np.sqrt(dt) * v * np.random.randn(1))
    plt.plot(p)
    for price in p:
      future_price.append(float(price))
    future.append(future_price)

JPM_America_Equity = pd.DataFrame()
for i in range(len(future)):
  JPM_America_Equity[i] = future[i]
print(JPM_America_Equity)

plt.grid()
plt.ylabel("Price")
```
![](https://i.imgur.com/JH8tQja.png)
```
JPM_America_Equity['mean'] = 0.0
JPM_America_Equity['q1'] = 0.0
JPM_America_Equity['q4'] = 0.0
for i in range(181):
  JPM_America_Equity['mean'][i] = sum(JPM_America_Equity.iloc[i])/10000
  tmp = []
  q1 = []
  q4 = []
  for k in JPM_America_Equity.iloc[i]:
    tmp.append(k)
    tmp.sort()
    q1 = tmp[0 : 2500]
    q4 = tmp[-2500:]
  JPM_America_Equity['q1'][i] = sum(q1)/2500
  JPM_America_Equity['q4'][i] = sum(q4)/2500

plt.figure(figsize = (7, 5))
plt.plot(JPM_America_Equity['mean'], label = 'mean')
plt.plot(JPM_America_Equity['q1'], label = 'pessimistic')
plt.plot(JPM_America_Equity['q4'], label = 'optimistic')
plt.legend(loc = 'best')
plt.grid()
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('JPM_America_Equity')
```
![](https://i.imgur.com/6Y17x8K.png)

### Future trend ( use future trend to re-construct portfolio )
![](https://i.imgur.com/MPV6GRU.png)


