import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 初始选股池
tickers = ['XLU', 'XLP', 'XLF', 'XLRE', 'XLE', 'XLV', 'XLI', # SPDR行业基金
 'QQQ', 'SPY', 'EFA', 'EEM', # 宽基ETF
 'GLD', 'LQD', 'IEF', 'TLT', # 另类资产ETF
 'PLTR', 'NVDA', 'MSFT', 
 'NFLX', 'NOW', 'AMD', 'MA', 'V', 'PANW', 'TSLA'] #高夏普比率股票 

# 1.数据获取 data collection & preprocess

# 1.1 获取收盘价 Close -> Return

# 计算收益率
data = yf.download(tickers, start='2024-04-01', end='2025-04-01')['Close']
returns = data.pct_change().dropna()
returns = returns.reset_index()
returns = returns.melt(id_vars='Date', var_name='Ticker', value_name='Return')

log_returns = np.log(data / data.shift(1)).dropna()
log_returns = log_returns.reset_index()
log_returns = log_returns.melt(id_vars='Date', var_name='Ticker', value_name='log_Return')

'''
# 极端值缩尾处理(optional）, 按当日所有股票收益率的分位数截取
def winsorize_day(group, lower=0.01, upper=0.01):
    return group.assign(
        Return=group['Return'].clip(
            lower=group['Return'].quantile(lower),
            upper=group['Return'].quantile(1 - upper)
        )
    )
returns = returns.groupby('Date', group_keys=False).apply(winsorize_day)
log_returns = log_returns.groupby('Date', group_keys=False).apply(winsorize_day)


# 存储
returns.to_csv("returns.csv")
log_returns.to_csv("log_returns.csv")
'''

# 1.2 获取无风险收益率 risk-free rate (3-month T-Bill rate)
import pandas_datareader.data as web
import datetime

start = datetime.datetime(2024, 4, 1)
end = datetime.datetime(2025, 4, 1)

t_bill = web.DataReader('DGS3MO', 'fred', start, end)
t_bill = t_bill.fillna(method='ffill')  # 用前值填补空缺日期
rf = t_bill['DGS3MO'] / 100 / 252  # 将 FRED 的年化百分比利率 → 日度
rf_df = rf.reset_index()
rf_df.columns = ['Date', 'rf']

# 合并1.1和1.2
merged_df1 = pd.merge(returns, rf_df, on='Date', how='left')
merged_df1.to_csv("excessive_returns.csv")
merged_df2 = pd.merge(log_returns, rf_df, on='Date', how='left')
merged_df2.to_csv("excessive_log_returns.csv")

'''
1.3 获取ETF基本信息(optional)
import yfinance as yf
import pandas as pd

fields = {
    'shortName': '名称',
    'category': '分类',
    'totalAssets': '资产规模 (美元)',
    'expenseRatio': '费率',
    'fundFamily': '基金公司',
    'inceptionDate': '成立时间',
    'ytdReturn': '今年以来回报',
    'beta': 'β系数'
}# 提取的字段

records = []
for symbol in tickers: # 获取所有信息
    try:
        info = yf.Ticker(symbol).info
        row = {'代码': symbol}
        for key, cn_name in fields.items():
            row[cn_name] = info.get(key)
        records.append(row)
    except Exception as e:
        print(f"无法获取 {symbol}：{e}")

etf_info = pd.DataFrame(records)
etf_info['资产规模 (美元)'] = etf_info['资产规模 (美元)'] / 100000000
etf_info = etf_info.rename(columns={'资产规模 (美元)': '资产规模 (亿美元)'})
'''

# 2. 计算各ETF的夏普比率和相关性 calculate Sharpe ratio & correlation

# 2.1 计算年化夏普比率

# 日度超额收益
merged = merged_df1.copy()
merged['excess_return'] = merged['Return'] - merged['rf']

# 日度超额收益均值和标准差
shp = merged.groupby('Ticker')['excess_return'] \
            .agg(mean_excess='mean', std_excess='std') \
            .reset_index()

# 年化夏普比率 = 日均超额收益 / 日度标准差 * sqrt(252)
shp['Sharpe'] = shp['mean_excess'] / shp['std_excess'] * np.sqrt(252)

# 输出结果
shp_sorted = shp.sort_values(by='Sharpe', ascending=False)
print(shp_sorted.set_index('Ticker')[['Sharpe']])
shp_sorted.set_index('Ticker')[['Sharpe']].to_csv("sharpe.csv")

# 2.2 计算相关性矩阵并画热力图

# 日度收益表 pivot 成宽表（使用原始日度 Return）
rets_wide = merged_df1.pivot(index='Date', columns='Ticker', values='Return')

# 按照夏普比率排序重新排列列
shp_order = shp_sorted['Ticker'].tolist()
rets_wide = rets_wide[shp_order]

# 计算相关性
corr = rets_wide.corr()

# 绘制热力图
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(corr.values, 
                interpolation='nearest', 
                aspect='auto', 
                cmap='coolwarm', 
                vmin=-1, vmax=1)

# 每个格子写上相关系数数值
for (i, j), val in np.ndenumerate(corr.values):
    ax.text(j, i, f"{val:.2f}", 
            ha='center', va='center', 
            fontsize=8, 
            color='white' if abs(val) > 0.5 else 'black')

# 坐标轴
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticklabels(corr.index)

# 标题
ax.set_title('Correlation Matrix of Selected Stocks (Past 1 Year)')
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Correlation Coefficient')

plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=1000, bbox_inches='tight')
plt.show()

# 3.最终选股结果并构建最大收益最小方差投资组合
stocks = merged_df1[merged_df1['Ticker'].isin(['XLF', 'XLE', 'QQQ', 'EFA', 'GLD', 'PLTR', 'NVDA', 'NFLX', 'NOW', 'V', 'PANW', 'TSLA'])]

from scipy.optimize import minimize
df_ret = stocks.pivot(index='Date', columns='Ticker', values='Return').sort_index()
rf = stocks[['Date','rf']].drop_duplicates().set_index('Date')['rf'].sort_index()

trading_days = 252
mu = df_ret.mean() * trading_days
Sigma = df_ret.cov() * trading_days
rf_ann = rf.mean() * trading_days

def neg_sharpe(weights, mu, Sigma, rf_ann): # 优化：最大化 (wᵀ·μ – rf_ann) / √(wᵀ·Σ·w)
    port_ret = weights @ mu
    port_vol = np.sqrt(weights @ Sigma @ weights)
    return - (port_ret - rf_ann) / port_vol

n = len(mu)
bounds = [(-1,1)] * n # 限制权重±100%
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}) # 权重之和等于1

w0 = np.ones(n) / n # 初始猜测：等权

res = minimize(neg_sharpe, w0,
               args=(mu, Sigma, rf_ann),
               method='SLSQP',
               bounds=bounds,
               constraints=cons)

w_opt = res.x
tickers = mu.index

port_ret_ann = w_opt @ mu
port_vol_ann = np.sqrt(w_opt @ Sigma @ w_opt)
sharpe_ann   = (port_ret_ann - rf_ann) / port_vol_ann

# 权重
print("Optimal weights:")
print(pd.Series(w_opt, index=tickers).round(4))
pd.Series(w_opt, index=tickers).round(4).to_csv("weights0.csv")

# 回测结果
r_p = df_ret.dot(w_opt)  # Daily portfolio return r_p(t)
excess_p = r_p - rf.reindex(r_p.index)  # Daily excess return: r_p(t) - rf(t)
cum_ret = (1 + r_p).prod()**(trading_days / len(r_p)) - 1  # Annualized return based on cumulative return

ann_vol_emp     = excess_p.std() * np.sqrt(trading_days)  # Annualized volatility
ann_ret_excess  = excess_p.mean() * trading_days          # Annualized excess return
sharpe_empirical = ann_ret_excess / ann_vol_emp           # Empirical Sharpe ratio

print(f"Annualized total return (including rf): {cum_ret:.2%}")
print(f"Annualized excess return: {ann_ret_excess:.2%}")
print(f"Annualized volatility: {ann_vol_emp:.2%}")
print(f"Empirical Sharpe ratio: {sharpe_empirical:.3f}")

# 绘制组合净值曲线
import matplotlib.pyplot as plt
(1 + r_p).cumprod().plot(title='Portfolio Net Value Curve', figsize=(8,4))
plt.ylabel('Cumulative Return')
plt.xlabel('Date')
plt.tight_layout()
plt.savefig("backtest_curve.png", dpi=1000, bbox_inches='tight')
plt.show()