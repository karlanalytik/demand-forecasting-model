---
title: "Predicting Retail Demand with Machine Learning"
subtitle: "Reporte Ejecutivo"
author: "Diana Treviño / Karla Soto"
jupyter: python3
format: 
  html:
    theme: pulse
    page-layout: full
    toc: true
    toc-title: Contents
    toc-depth: 3
    toc-expand: 2
    toc-location: left
    df-print: kable
---

## 1. Contexto

Este proyecto desarrolla un modelo de *demand forecasting* para predecir ventas mensuales por tienda–producto, usando datos históricos de ventas.

El objetivo principal es construir un pipeline reproducible que permita:
- Generar features temporales.
- Entrenar modelos baseline.
- Producir predicciones para el conjunto test.

---

## 2. Datos

- Fuente: historial de ventas mensuales.
- Granularidad: tienda–producto–mes.
- Variable objetivo: `item_cnt_month`.

---

## 3. Exploratory Data Analysis (EDA)

Hallazgos principales:

- Distribución altamente sesgada de ventas (muchos ceros).
- Presencia de outliers → se aplicó *clipping*.
- Alta estacionalidad mensual.
- Variabilidad significativa entre tiendas y productos.

---

## 4. Feature Engineering

Se generaron:

- Lags de ventas:
  - `item_cnt_month_clipped_lag_1`
  - `item_cnt_month_clipped_lag_2`
  - `item_cnt_month_clipped_lag_3`
  - `item_cnt_month_clipped_lag_6`
  - `item_cnt_month_clipped_lag_12`

- Agregados:
  - `shop_month_mean_lag_1`
  - `item_month_mean_lag_1`

Se cuidó que:
- Train y test tuvieran exactamente las mismas columnas.
- Valores faltantes fueran imputados en 0.
- No hubiera fugas de información futura.

---

## 5. Modelo Baseline

Se entrenó un modelo **Ridge Regression** como baseline:

- Ventajas:
  - Rápido.
  - Estable.
  - Buen punto de comparación inicial.

- Inputs:
  - Lags de ventas.
  - Promedios por tienda y producto.

---

## 6. Pipeline Reproducible

El flujo end-to-end queda:

1. `01_eda.ipynb`  
   → análisis exploratorio.

2. `02_features.ipynb`  
   → generación de features y lags.

3. `03_train.ipynb`  
   → entrenamiento + predicción + `submission.csv`.

Esto permite regenerar todo el pipeline desde cero.

---

## 7. Resultados

- Pipeline ejecutable end-to-end.
- Features consistentes entre train y test.
- Generación exitosa de predicciones.
- Archivo `submission.csv` producido correctamente.

---

## 8. Próximos Pasos

- Tuning de hiperparámetros.
- Nuevas features agregadas.
- Modelos alternativos (LightGBM, XGBoost).
- Validación temporal.
- Ensembling.

---

## 9. Conclusión

El proyecto establece una base sólida y reproducible para *demand forecasting*.  
El enfoque incremental permitirá mejoras sustanciales en desempeño con modelos más avanzados y mejores features.
