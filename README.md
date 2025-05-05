# Quant Trader Optimization — Electric Power Markets

Este proyecto presenta una estrategia cuantitativa para optimizar un portafolio de compras y ventas en el mercado eléctrico, utilizando datos históricos hasta el día d−2 y respetando un presupuesto máximo de USD $50,000 por día. El portafolio final fue validado durante más de 11 meses y alcanzó un ROI superior al 215%.

---

##  Estructura del Proyecto

```
📁 /
├── dataset_pt_20250428v0.csv            # Dataset original (no incluido en este repo por tamaño)
├── QuantTrading.xlsx         # Resultados del portafolio optimizado
├── cumulative_return_versionB.png       # Gráfica de retorno acumulado
├── QuantTradingv1.py           # Código principal (estrategia optimizada)
└── README.md                            # Este archivo
```

---

## Instrucciones de instalación

Este proyecto requiere Python 3.10+ y las siguientes librerías:

### 1. Crear entorno virtual (opcional pero recomendado)
```bash
python -m venv venv
source venv/bin/activate      # En Mac/Linux
venv\Scripts\activate         # En Windows
```

### 2. Instalar dependencias
```bash
pip install pandas numpy matplotlib seaborn openpyxl
```

---

## Cómo ejecutar el proyecto

1. Descarga este repositorio
2. Asegúrate de tener el archivo `dataset_pt_20250428v0.csv` en la carpeta raíz, tambien se puede usar filepath
3. Ejecuta el script `QuantTrading.py` desde tu IDE (VS Code o Spyder)

El script generará:
- Un archivo Excel con el portafolio diario y resumen de métricas (`QuantOptimized_VersionB.xlsx`)
- Una gráfica de retorno acumulado (`cumulative_return_versionB.png`)
- Métricas clave impresas en consola

---

## Resultados Esperados

**ROI:** 215.81%  
**Sharpe Ratio:** 0.193  
**Max Drawdown:** USD $530,338.58  
**Periodo Validado:** 335 días

---

## Requisitos de la prueba cumplidos

- Uso exclusivo de datos hasta d−2
- Inversión diaria limitada a USD $50,000
- Portafolio con al menos 5 nodos por día
- Retornos expresados en términos financieros reales
- Ventana móvil para evitar overfitting
- Reporte con métricas, gráficas y recomendaciones
- Analisis de las graficas
---

## Recomendaciones futuras

- Incorporar modelos predictivos supervisados (regresión o árboles)
- Introducir límites dinámicos de riesgo según drawdown diario
- Ampliar el portafolio con estrategias alternativas de cobertura

---

## Contacto

Jennifer Lopez   
jennifer.lpzch@gmail.com 
