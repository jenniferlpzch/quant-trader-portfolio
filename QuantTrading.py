
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Load dataset
df = pd.read_csv('/Users/jenniferlopez/Downloads/prueba_tecnica (1) (1)/dataset_pt_20250428v0.csv', parse_dates=["DateTime"])

# Clean missing prices
df['DAMSPP'] = df.groupby('AssetID')['DAMSPP'].transform(lambda x: x.ffill().bfill())
df['RTMSPP'] = df.groupby('AssetID')['RTMSPP'].transform(lambda x: x.ffill().bfill())


# Separate BID and OFFER
bids = df[df['TransactionType'] == 'BID'].copy()
offers = df[df['TransactionType'] == 'OFFER'].copy()

bids['BidPrice'] = bids['BidPrice'].fillna(bids['DAMSPP'])
bids['MWBid'] = bids['MWBid'].fillna(1).astype(int)
offers['OfferPrice'] = offers['OfferPrice'].fillna(offers['RTMSPP'])
offers['MWOffer'] = offers['MWOffer'].fillna(1).astype(int)

# Calculate returns
bids['Cost'] = bids['DAMSPP'] * bids['MWBid']
bids['Income'] = bids['RTMSPP'] * bids['MWBid']
bids['Return'] = bids['Income'] - bids['Cost']

offers['Cost'] = offers['RTMSPP'] * offers['MWOffer']
offers['Income'] = offers['DAMSPP'] * offers['MWOffer']
offers['Return'] = offers['Income'] - offers['Cost']

df = pd.concat([bids, offers])
df = df.sort_values('DateTime').reset_index(drop=True)
df['Date'] = df['DateTime'].dt.date

# Establecer fechas para entrenamiento y validación (ajustables)
train_end_date = datetime.date(2024, 08, 31)   # 8 meses desde enero
valid_start_date = train_end_date + datetime.timedelta(days=1)
valid_end_date = datetime.date(2025, 1, 15)    # validación hasta 15 de enero 2025

# Apply d-2 constraint
min_date = df['Date'].min() + datetime.timedelta(days=62)  # d-2 + 60-day window
valid_df = df[df['Date'] >= min_date]

# Strategy parameters
daily_budget = 50000
portfolio = []

for current_date in sorted(valid_df['Date'].unique()):
    history_cutoff = current_date - datetime.timedelta(days=2)
    window_start = history_cutoff - datetime.timedelta(days=60)
    history = df[(df['Date'] > window_start) & (df['Date'] <= history_cutoff)]

    if history.empty:
        continue

    stats = history.groupby('AssetID').agg(
        mean_return=('Return', 'mean'),
        std_return=('Return', 'std'),
        avg_cost=('Cost', 'mean')
    ).dropna()

    stats = stats[stats['std_return'] > 0]
    stats['sharpe_score'] = stats['mean_return'] / stats['std_return']
    stats = stats[stats['sharpe_score'] > 0]  # exclude poor risk-adjusted nodes
    stats['roi_score'] = stats['mean_return'] / stats['avg_cost']

    top_assets = stats.sort_values(by='roi_score', ascending=False).head(10).index.tolist()
    candidates = valid_df[(valid_df['Date'] == current_date) & (valid_df['AssetID'].isin(top_assets))].copy()
    if candidates.empty:
        continue

    candidates = candidates.merge(stats[['roi_score']], left_on='AssetID', right_index=True)
    total_score = candidates['roi_score'].sum()
    if total_score == 0:
        continue

    candidates['weight'] = candidates['roi_score'] / total_score
    candidates['allocated_capital'] = daily_budget * candidates['weight']

    factor = candidates[['MWBid', 'MWOffer']].fillna(1).min(axis=1)
    candidates['MW'] = np.floor(candidates['allocated_capital'] / candidates['Cost']) * factor

    # Avoid NaN errors
    candidates = candidates.replace([np.inf, -np.inf], np.nan)
    candidates = candidates.dropna(subset=['MW'])
    candidates['MW'] = candidates['MW'].clip(lower=1).astype(int)

    candidates['ActualCost'] = candidates['Cost'] / factor * candidates['MW']
    candidates['ActualReturn'] = candidates['Return'] / factor * candidates['MW']

    daily_trades = candidates[candidates['ActualCost'] <= daily_budget]
    if len(set(daily_trades['AssetID'])) >= 5:
        for _, row in daily_trades.iterrows():
            portfolio.append({
                'Date': current_date,
                'AssetID': row['AssetID'],
                'TransactionType': row['TransactionType'],
                'MW': row['MW'],
                'Return': row['ActualReturn'],
                'Cost': row['ActualCost']
            })

# Results
portfolio_df = pd.DataFrame(portfolio)
portfolio_df["Date"] = pd.to_datetime(portfolio_df["Date"])
daily_returns = portfolio_df.groupby("Date")["Return"].sum()
cumulative_return = daily_returns.cumsum()
roll_max = cumulative_return.cummax()
drawdown = roll_max - cumulative_return

# Performance metrics
total_return = portfolio_df["Return"].sum()
total_cost = portfolio_df["Cost"].sum()
roi = total_return / total_cost
max_drawdown = drawdown.max()
mean_return = daily_returns.mean()
std_return = daily_returns.std()
sharpe_ratio = mean_return / std_return if std_return > 0 else float("nan")
validation_start = portfolio_df["Date"].min().date()
validation_end = portfolio_df["Date"].max().date()
validation_days = (portfolio_df["Date"].max() - portfolio_df["Date"].min()).days

# Export, could use standard, instead I used the file my filepath
output_path = "/Users/jenniferlopez/Downloads/QuantOptimized_VersionB.xlsx"
portfolio_df.to_excel(output_path, sheet_name="Portfolio", index=False)

summary_df = pd.DataFrame({
    "Valor": [
        f"${total_return:,.2f}",
        f"${total_cost:,.2f}",
        f"{roi:.2%}",
        f"{sharpe_ratio:.3f}",
        f"${max_drawdown:,.2f}",
        f"{validation_start} a {validation_end} ({validation_days} días)"
    ]
}, index=[
    "Retorno total (USD)",
    "Capital invertido (USD)",
    "ROI",
    "Sharpe Ratio",
    "Max Drawdown",
    "Periodo de validación"
])

with pd.ExcelWriter(output_path, engine="openpyxl", mode="a") as writer:
    summary_df.to_excel(writer, sheet_name="Performance Summary")

# Plot cumulative return
plt.figure(figsize=(10, 5))
sns.lineplot(x=cumulative_return.index, y=cumulative_return.values)
plt.title("Cumulative Return Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/jenniferlopez/Downloads/cumulative_return_versionB.png")
plt.show()
roll_max = cumulative_return.cummax()
drawdown = roll_max - cumulative_return

# Graficar drawdown
plt.figure(figsize=(12, 5))
sns.lineplot(x=drawdown.index, y=drawdown.values)
plt.title("Drawdown Over Time")
plt.xlabel("Date")
plt.ylabel("Drawdown (USD)")
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/jenniferlopez/Downloads/drawdown_over_time.png")
plt.show()

# Print clean summary
print("\n RESUMEN DE MÉTRICAS (Versión B)")
print(f"Retorno total (USD): ${total_return:,.2f}")
print(f"Capital invertido (USD): ${total_cost:,.2f}")
print(f"ROI: {roi:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"Max Drawdown (USD): ${max_drawdown:,.2f}")
print(f"Periodo de validación: {validation_start} a {validation_end} ({validation_days} días)")

