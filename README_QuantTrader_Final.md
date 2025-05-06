
# ⚡ Quant Trader Portfolio Optimization – Validación Cruzada

Este proyecto simula una estrategia cuantitativa para construir un portafolio diario en el mercado eléctrico. A través de datos históricos y métricas de rendimiento, se seleccionan nodos de energía con el objetivo de maximizar retornos bajo restricciones operativas reales.

---

## Objetivo

- Optimizar la asignación diaria de capital (máx. USD 50,000)
- Seleccionar al menos 5 nodos por día
- Calcular y evaluar métricas clave como ROI, Sharpe Ratio y Drawdown
- Evitar el uso de datos futuros mediante validación cruzada temporal
- Exportar resultados y análisis para revisión técnica

## Empieza AQUI, nota antes de comenzar
- Antes de comenzar, es importante saber que el archivo QuantTrading.py es el version con Cambios
- Hechos, QuantTradingv1.py es el original, donde luego se adapto cambios para mejorar resultados
- Este de QuantValidacionCruzada.py es el version con cambios hechos con validacion, version final
- Para mi es importante reconocer y ver los adaptaciones hechos, para ver que a mejorado.

---

## Validación cruzada (out-of-sample)

Para asegurar una evaluación realista del portafolio, se implementó una validación cruzada temporal:

| Etapa         | Periodo                      |
|---------------|------------------------------|
| Entrenamiento | Enero 2024 – Agosto 2024     |
| Validación    | Septiembre 2024 – Octubre 2024 |

Durante la validación, las decisiones de inversión se toman usando exclusivamente datos históricos hasta el día `d−2`, replicando condiciones reales del mercado eléctrico.

---

## Cambios y mejoras implementadas

-  Separación clara entre entrenamiento y validación (validación cruzada)
-  Selección dinámica de nodos basada en métricas ROI y Sharpe
-  Control estricto del presupuesto diario (`USD 50,000`)
-  Exportación a Excel con múltiples hojas: `Portfolio`, `Performance Summary`, `Recommendations`
-  Visualización del retorno acumulado (`return_validacion_cruzada.png`) con `plt.show()` habilitado
-  Inclusión de análisis gráfico y recomendaciones
-  Código limpio, modular y comentado
-  Publicación del proyecto completo en GitHub

---

## Fragmento clave del código

```python
# Definir periodos de entrenamiento y validación cruzada
train_end_date = datetime.date(2024, 8, 31)
valid_start_date = train_end_date + datetime.timedelta(days=1)
valid_end_date = datetime.date(2025, 1, 15)

train_df = df[df['Date'] <= train_end_date]
valid_df = df[(df['Date'] >= valid_start_date) & (df['Date'] <= valid_end_date)]

portfolio_df = simulate_portfolio(train_df, valid_df)
summary, cumulative_return, cleaned_portfolio = calculate_metrics(portfolio_df)
export_outputs(cleaned_portfolio, summary, cumulative_return)
```

---

## Resultados finales (Validación)

| Métrica              | Valor                 |
|----------------------|------------------------|
| Periodo validado     | Sept 1 – Oct 31, 2024 (60 días) |
| Retorno total (USD)  | $976,166.64            |
| Capital invertido    | $2,873,693.48          |
| ROI                  | 33.97%                 |
| Sharpe Ratio         | 0.138                  |
| Máximo Drawdown      | $170,419.13            |

---

## Archivos generados

| Archivo                          | Descripción                                        |
|----------------------------------|----------------------------------------------------|
| `QuantTrading_ValidacionCruzada.py` | Código principal con validación cruzada aplicada   |
| `QuantValidacionCruzada.xlsx`   | Resultados: hoja de portafolio, métricas, recomendaciones |
| `return_validacion_cruzada.png` | Gráfico del retorno acumulado                      |
| `Analisis mercado energia.pdf`  | Reporte técnico con análisis, gráficas y mejoras   |

---

## 💡 Recomendaciones incluidas

- Diversificar nodos con menor volatilidad
- Incorporar reglas de control de pérdidas (“cut-loss”)
- Mejorar consistencia del retorno diario post-spike
- Ampliar la selección de nodos si el capital no se utiliza por completo

---

## ⚙️ Ejecución

Instala dependencias:

```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

Ejecuta el script:

```bash
python QuantTrading_ValidacionCruzada.py
```

Se generará:
- Un Excel con resultados en 3 hojas
- Gráfico PNG del retorno acumulado
- Resumen en consola

---

## Notas finales

Este proyecto cumple con todos los requerimientos técnicos establecidos por el reto, evitando el uso de datos futuros y validando fuera de muestra con reglas realistas del mercado eléctrico. Se entrega documentación completa, código funcional y análisis crítico del desempeño.

---


