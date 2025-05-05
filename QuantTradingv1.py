# quant_trader_project.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Load the dataset
df = pd.read_csv('/Users/jenniferlopez/Downloads/prueba_tecnica (1) (1)/dataset_pt_20250428v0.csv', parse_dates=["DateTime"])

# Fill missing prices by forward/backward fill within each node
df['DAMSPP'] = df.groupby('AssetID')['DAMSPP'].transform(lambda x: x.ffill().bfill())
df['RTMSPP'] = df.groupby('AssetID')['RTMSPP'].transform(lambda x: x.ffill().bfill())

# Separate BID and OFFER transactions
bids = df[df['TransactionType'] == 'BID'].copy()
offers = df[df['TransactionType'] == 'OFFER'].copy()

# Fill missing bid values
bids['BidPrice'] = bids['BidPrice'].fillna(bids['DAMSPP'])
bids['MWBid'] = bids['MWBid'].fillna(1).astype(int)

# Fill missing offer values
offers['OfferPrice'] = offers['OfferPrice'].fillna(offers['RTMSPP'])
offers['MWOffer'] = offers['MWOffer'].fillna(1).astype(int)

# Compute cost, income, return
bids['Cost'] = bids['DAMSPP'] * bids['MWBid']
bids['Income'] = bids['RTMSPP'] * bids['MWBid']
bids['Return'] = bids['Income'] - bids['Cost']

offers['Cost'] = offers['RTMSPP'] * offers['MWOffer']
offers['Income'] = offers['DAMSPP'] * offers['MWOffer']
offers['Return'] = offers['Income'] - offers['Cost']

# Combine cleaned datasets
clean_df = pd.concat([bids, offers], axis=0).sort_values('DateTime').reset_index(drop=True)
clean_df['Date'] = clean_df['DateTime'].dt.date

# Filter for dates with full historical data (d-2 rule)
min_date = clean_df['Date'].min() + datetime.timedelta(days=2)
valid_df = clean_df[clean_df['Date'] >= min_date]

# Simulate daily trading strategy
daily_budget = 50000
portfolio = []

for current_date in sorted(valid_df['Date'].unique()):
    history_cutoff = current_date - datetime.timedelta(days=2)
    history = clean_df[clean_df['Date'] <= history_cutoff]

    if history.empty:
        continue

    # Select top 10 profitable nodes
    top_assets = history.groupby('AssetID')['Return'].mean().sort_values(ascending=False).head(10).index.tolist()
    candidates = valid_df[(valid_df['Date'] == current_date) & (valid_df['AssetID'].isin(top_assets))]
    candidates = candidates.sort_values(by='Return', ascending=False)

    spent = 0
    daily_trades = []

    for _, row in candidates.iterrows():
        mw = row['MWBid'] if row['TransactionType'] == 'BID' else row['MWOffer']
        cost = row['Cost']

        if spent + cost <= daily_budget:
            daily_trades.append({
                'Date': current_date,
                'AssetID': row['AssetID'],
                'TransactionType': row['TransactionType'],
                'Return': row['Return'],
                'Cost': row['Cost']
            })
            spent += cost

        if spent >= daily_budget:
            break

    portfolio.extend(daily_trades)

# Convert to DataFrame
portfolio_df = pd.DataFrame(portfolio)

# Compute performance metrics
daily_returns = portfolio_df.groupby('Date')['Return'].sum()
cumulative_return = daily_returns.cumsum()
roi = portfolio_df['Return'].sum() / portfolio_df['Cost'].sum()

# Print results
print("Total Return (USD):", portfolio_df['Return'].sum())
print("Total Invested (USD):", portfolio_df['Cost'].sum())
print("Return on Investment (ROI): {:.2%}".format(roi))

# Plot cumulative return
plt.figure(figsize=(10, 6))
sns.lineplot(x=cumulative_return.index, y=cumulative_return.values)
plt.title("Cumulative Return Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("cumulative_return_plot.png")
plt.show()

daily_returns = portfolio_df.groupby('Date')['Return'].sum()
cumulative_return = daily_returns.cumsum()

# Drawdown
roll_max = cumulative_return.cummax()
drawdown = roll_max - cumulative_return
max_drawdown = drawdown.max()

roi = portfolio_df['Return'].sum() / portfolio_df['Cost'].sum()
total_return = portfolio_df['Return'].sum()
total_cost = portfolio_df['Cost'].sum()

# Print results
print("Total Return (USD):", f"${total_return:,.2f}")
print("Total Invested (USD):", f"${total_cost:,.2f}")
print("Return on Investment (ROI): {:.2%}".format(roi))
print("Max Drawdown (USD):", f"${max_drawdown:,.2f}")

# Plot cumulative return
plt.figure(figsize=(10, 6))
sns.lineplot(x=cumulative_return.index, y=cumulative_return.values)
plt.title("Cumulative Return Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("cumulative_return_plot.png")
plt.show()

# Plot drawdown
plt.figure(figsize=(10, 6))
sns.lineplot(x=drawdown.index, y=drawdown.values)
plt.title("Drawdown Over Time")
plt.xlabel("Date")
plt.ylabel("Drawdown (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("drawdown_plot.png")
plt.show()

# Save portfolio results
portfolio_df.to_csv("quant_trader_portfolio_results.csv", index=False)

# Add recommendations section
recommendations = [
    "Allocate more capital to historically less volatile nodes.",
    "Prioritize nodes with consistent Sharpe performance over time.",
    "Scale down capital when recent volatility spikes (dynamic budget).",
    "Expand node selection beyond top 10 if too many days are excluded.",
    "Consider adding time-lagged features to predict return behavior."
]
recommendations_df = pd.DataFrame(recommendations, columns=["Recommendations"])
recommendations_df.to_csv("quant_trader_recommendations.csv", index=False)

