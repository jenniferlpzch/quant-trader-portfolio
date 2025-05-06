#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 20:15:07 2025

@author: jenniferlopez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

# Parámetros globales
DAILY_BUDGET = 50000
WINDOW_DAYS = 60
#This is where my data is stored, change to where ever your data may be
INPUT_PATH = "/Users/jenniferlopez/Downloads/prueba_tecnica (1) (1)/dataset_pt_20250428v0.csv"
OUTPUT_EXCEL = "/Users/jenniferlopez/Downloads/QuantValidacionCruzada.xlsx"
OUTPUT_PLOT = "/Users/jenniferlopez/Downloads/return_validacion_cruzada.png"

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=["DateTime"])
    df['DAMSPP'] = df.groupby('AssetID')['DAMSPP'].transform(lambda x: x.ffill().bfill())
    df['RTMSPP'] = df.groupby('AssetID')['RTMSPP'].transform(lambda x: x.ffill().bfill())
    df['Date'] = df['DateTime'].dt.date

    bids = df[df['TransactionType'] == 'BID'].copy()
    offers = df[df['TransactionType'] == 'OFFER'].copy()

    bids['BidPrice'] = bids['BidPrice'].fillna(bids['DAMSPP'])
    bids['MWBid'] = bids['MWBid'].fillna(1).astype(int)
    offers['OfferPrice'] = offers['OfferPrice'].fillna(offers['RTMSPP'])
    offers['MWOffer'] = offers['MWOffer'].fillna(1).astype(int)

    bids['Cost'] = bids['DAMSPP'] * bids['MWBid']
    bids['Income'] = bids['RTMSPP'] * bids['MWBid']
    bids['Return'] = bids['Income'] - bids['Cost']

    offers['Cost'] = offers['RTMSPP'] * offers['MWOffer']
    offers['Income'] = offers['DAMSPP'] * offers['MWOffer']
    offers['Return'] = offers['Income'] - offers['Cost']

    return pd.concat([bids, offers]).sort_values('DateTime').reset_index(drop=True)

def simulate_portfolio(train_df, valid_df):
    portfolio = []

    for current_date in sorted(valid_df['Date'].unique()):
        history_cutoff = current_date - datetime.timedelta(days=2)
        window_start = history_cutoff - datetime.timedelta(days=WINDOW_DAYS)
        history = train_df[(train_df['Date'] > window_start) & (train_df['Date'] <= history_cutoff)]

        if history.empty:
            continue

        stats = history.groupby('AssetID').agg(
            mean_return=('Return', 'mean'),
            std_return=('Return', 'std'),
            avg_cost=('Cost', 'mean')
        ).dropna()

        stats = stats[stats['std_return'] > 0]
        stats['sharpe_score'] = stats['mean_return'] / stats['std_return']
        stats = stats[stats['sharpe_score'] > 0]
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
        candidates['allocated_capital'] = DAILY_BUDGET * candidates['weight']
        factor = candidates[['MWBid', 'MWOffer']].fillna(1).min(axis=1)
        candidates['MW'] = np.floor(candidates['allocated_capital'] / candidates['Cost']) * factor

        candidates = candidates.replace([np.inf, -np.inf], np.nan)
        candidates = candidates.dropna(subset=['MW'])
        candidates['MW'] = candidates['MW'].clip(lower=1).astype(int)
        candidates['ActualCost'] = candidates['Cost'] / factor * candidates['MW']
        candidates['ActualReturn'] = candidates['Return'] / factor * candidates['MW']

        daily_trades = candidates[candidates['ActualCost'] <= DAILY_BUDGET]
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

    return pd.DataFrame(portfolio)

def calculate_metrics(portfolio_df):
    portfolio_df["Date"] = pd.to_datetime(portfolio_df["Date"])
    daily_returns = portfolio_df.groupby("Date")["Return"].sum()
    cumulative_return = daily_returns.cumsum()
    roll_max = cumulative_return.cummax()
    drawdown = roll_max - cumulative_return

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

    summary = {
        "Retorno total (USD)": f"${total_return:,.2f}",
        "Capital invertido (USD)": f"${total_cost:,.2f}",
        "ROI": f"{roi:.2%}",
        "Sharpe Ratio": round(sharpe_ratio, 3),
        "Max Drawdown": f"${max_drawdown:,.2f}",
        "Periodo de validación": f"{validation_start} a {validation_end} ({validation_days} días)"
    }

    return summary, cumulative_return, portfolio_df

def export_outputs(portfolio_df, summary_dict, cumulative_return):
    summary_df = pd.DataFrame.from_dict(summary_dict, orient='index', columns=["Valor"])

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        portfolio_df.to_excel(writer, sheet_name="Portfolio", index=False)
        summary_df.to_excel(writer, sheet_name="Performance Summary")

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=cumulative_return.index, y=cumulative_return.values)
    plt.title("Cumulative Return Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)  # Guarda el gráfico como imagen
    plt.show()                # Muestra el gráfico en Spyder
    plt.close()               # Cierra la figura para liberar memoria

    

    print("RESUMEN DE MÉTRICAS con validacion cruzada)")
    for k, v in summary_dict.items():
        print(f"{k}: {v}")

# === MAIN ===


df = load_and_prepare_data(INPUT_PATH)

# Último día del conjunto de entrenamiento (historial conocido)
# Usamos todos los datos anteriores a esta fecha para calcular métricas (Sharpe, ROI, etc.)
# El periodo de validación (o testeo fuera de muestra) empieza un día después del entrenamiento
# En este rango vamos a simular decisiones reales sin reentrenar ni recalcular
# Fecha final del periodo de validación
# Todo lo que ocurra después de esta fecha no se considera en la simulación
# Definir periodos de entrenamiento y validación
# Filtramos los datos de entrenamiento: solo incluyen fechas hasta el 31 de agosto de 2024
# Estos datos se usarán para calcular estadísticas históricas
train_end_date = datetime.date(2024, 8, 31)
valid_start_date = train_end_date + datetime.timedelta(days=1)
valid_end_date = datetime.date(2025, 1, 15)
# Filtramos los datos de validación: de septiembre 2024 a enero 2025
# Aquí se aplican las decisiones de inversión simuladas
 #Simulamos el portafolio: para cada día del periodo de validación,
# usamos únicamente información pasada (hasta d−2) proveniente del periodo de entrenamiento
train_df = df[df['Date'] <= train_end_date]
valid_df = df[(df['Date'] >= valid_start_date) & (df['Date'] <= valid_end_date)]
# Calculamos las métricas clave de desempeño: ROI, drawdown, Sharpe, etc.
portfolio_df = simulate_portfolio(train_df, valid_df)
summary, cumulative_return, cleaned_portfolio = calculate_metrics(portfolio_df)
export_outputs(cleaned_portfolio, summary, cumulative_return)
