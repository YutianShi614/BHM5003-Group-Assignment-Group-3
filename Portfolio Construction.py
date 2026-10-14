
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize


tickers = {
    '粤海投资': '0270.HK',
    '绿色动力环保': '1330.HK',
    '电能实业': '0006.HK',
    '华润电力': '0836.HK',
    '中电控股': '0002.HK',
    '宁德时代': '3750.HK',
}

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# ---------- 1. 下载数据 ----------
price_dict = {}
return_dict = {}

for name, ticker_symbol in tickers.items():
    df = yf.download(ticker_symbol, start=start_date, end=end_date,
                     auto_adjust=True, progress=False)

    if df.empty:
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if 'Close' not in df.columns:
        continue

    price = df['Close'].copy()
    price.index = pd.to_datetime(price.index).date
    price = price.dropna()

    daily_return = price.pct_change()

    price_dict[name] = price
    return_dict[name] = daily_return

price_df = pd.DataFrame(price_dict)
return_df = pd.DataFrame(return_dict)

price_df.index.name = 'Date'
return_df.index.name = 'Date'

returns = return_df.dropna()

# ---------- 2. 基准 ----------
bench = yf.download('^HSI', start=start_date, end=end_date,
                    auto_adjust=True, progress=False)
if isinstance(bench.columns, pd.MultiIndex):
    bench.columns = bench.columns.get_level_values(0)

bench_price = bench['Close'].copy()
bench_price.index = pd.to_datetime(bench_price.index).date
bench_return = bench_price.pct_change()

aligned = pd.concat([returns, bench_return.rename('Benchmark')], axis=1).dropna()
stock_returns = aligned[returns.columns]
bench_returns = aligned['Benchmark']

# ---------- 3. 每只股票指标 ----------
TRADING_DAYS = 252
RISK_FREE = 0.03

stock_metrics = []
for name in stock_returns.columns:
    r = stock_returns[name]
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RISK_FREE) / ann_vol

    cov = np.cov(r, bench_returns)[0, 1]
    var_b = np.var(bench_returns, ddof=1)
    beta = cov / var_b

    stock_metrics.append({
        'Stock': name,
        'Annualized Return': ann_ret,
        'Annualized Volatility': ann_vol,
        'Beta': beta,
        'Sharpe Ratio': sharpe,
    })

stock_metrics_df = pd.DataFrame(stock_metrics).set_index('Stock')

# ---------- 4. 等权重组合 ----------
n = len(stock_returns.columns)

port_ret_eq = stock_returns.mean(axis=1)
ann_ret_eq = port_ret_eq.mean() * TRADING_DAYS
ann_vol_eq = port_ret_eq.std() * np.sqrt(TRADING_DAYS)
sharpe_eq = (ann_ret_eq - RISK_FREE) / ann_vol_eq

equal_weight_df = pd.DataFrame({
    'Metric': ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio'],
    'Value': [ann_ret_eq, ann_vol_eq, sharpe_eq],
}).set_index('Metric')

# ---------- 5. 最大 Sharpe 优化组合 ----------
mean_returns = stock_returns.mean() * TRADING_DAYS
cov_matrix = stock_returns.cov() * TRADING_DAYS

def neg_sharpe(weights):
    port_ret = np.sum(mean_returns * weights)
    port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
    return -(port_ret - RISK_FREE) / port_vol

constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
bounds = [(0.05, 1)] * n   # 下限 5%
init_weights = np.ones(n) / n

result = minimize(neg_sharpe, init_weights, method='SLSQP',
                  bounds=bounds, constraints=constraints)

opt_weights = result.x
opt_ret = np.sum(mean_returns * opt_weights)
opt_vol = np.sqrt(opt_weights @ cov_matrix.values @ opt_weights)
opt_sharpe = (opt_ret - RISK_FREE) / opt_vol

optimal_df = pd.DataFrame({
    'Metric': ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio'],
    'Value': [opt_ret, opt_vol, opt_sharpe],
}).set_index('Metric')

optimal_weights_df = pd.DataFrame({
    'Stock': stock_returns.columns,
    'Weight': opt_weights,
}).set_index('Stock')

# ---------- 6. Correlation ----------
correlation_df = stock_returns.corr()

# ---------- 7. 写入 Excel ----------
excel_filename = 'green_portfolio_6stocks_1y.xlsx'

with pd.ExcelWriter(excel_filename) as writer:
    price_df.to_excel(writer, sheet_name='Adj Close')
    return_df.to_excel(writer, sheet_name='Daily Return')

    row = 0

    # 表1：每只股票指标
    stock_metrics_df.round(6).to_excel(writer, sheet_name='Metrics', startrow=row)
    row += len(stock_metrics_df) + 4

    # 表2：等权重组合
    equal_weight_df.round(6).to_excel(writer, sheet_name='Metrics', startrow=row)
    row += len(equal_weight_df) + 4

    # 表3：最优组合
    optimal_df.round(6).to_excel(writer, sheet_name='Metrics', startrow=row)
    row += len(optimal_df) + 4

    # 表4：最优权重
    optimal_weights_df.round(6).to_excel(writer, sheet_name='Metrics', startrow=row)
    row += len(optimal_weights_df) + 4

    # 表5：Correlation
    correlation_df.round(6).to_excel(writer, sheet_name='Metrics', startrow=row)