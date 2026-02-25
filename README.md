# 📈 Sales Forecasting Dashboard — Noviembre 2025

Dashboard interactivo de predicción de ventas para una tienda de artículos deportivos, construido con **Streamlit** y un modelo **HistGradientBoostingRegressor** de scikit-learn.

## 🚀 ¿Qué hace este proyecto?

A partir de datos históricos de ventas (productos, precios, competencia, fechas especiales…), el sistema:

1. **Entrena** un modelo de machine learning con ingeniería de características avanzada (lags, medias móviles, variables temporales, codificación OHE…).
2. **Prepara** los datos de inferencia replicando exactamente el mismo pipeline de transformación.
3. **Predice** las ventas diarias de noviembre 2025 de forma **recursiva** (cada día usa las predicciones anteriores como lag).
4. **Visualiza** los resultados en un dashboard oscuro e interactivo con escenarios de competencia y filtros por producto.

---

## 🗂️ Estructura del proyecto

```
Data_Practica/
├── app/
│   ├── app.py                  # Dashboard principal de Streamlit
│   └── main.py                 # Punto de entrada (launcher)
├── data/
│   └── processed/              # Datos de inferencia transformados (CSV)
├── docs/                       # Documentación adicional
├── models/                     # Modelo entrenado (.joblib)
├── notebooks/
│   ├── entrenamiento.ipynb     # EDA, feature engineering y entrenamiento
│   └── Forecasting.ipynb       # Notebook de inferencia y predicción
├── scripts/
│   ├── prepare_inference.py    # Prepara y transforma datos de inferencia
│   ├── run_forecasting.py      # Ejecuta las predicciones y guarda resultados
│   └── verify_consistency.py  # Verifica consistencia de columnas entre train/infer
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Instalación

```bash
# 1. Clona el repositorio
git clone https://github.com/tu_usuario/Data_Practica.git
cd Data_Practica

# 2. Crea y activa un entorno virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate   # Windows

# 3. Instala las dependencias
pip install -r requirements.txt
```

---

## ▶️ Uso

### 1. Preparar los datos de inferencia

```bash
python scripts/prepare_inference.py
```

### 2. (Opcional) Verificar consistencia de columnas

```bash
python scripts/verify_consistency.py
```

### 3. Lanzar el dashboard

```bash
streamlit run app/app.py
```

---

## 🧠 Modelo

| Parámetro | Valor |
|---|---|
| Algoritmo | `HistGradientBoostingRegressor` |
| Características | ~90 features (lags, medias móviles, OHE, temporales…) |
| Predicción | Recursiva día a día (noviembre 2025) |
| Archivo | `models/modelo_final_forecasting.joblib` |

---

## 📊 Funcionalidades del Dashboard

- **Selector de producto** — 25 productos deportivos (running, fitness, outdoor, wellness)
- **Ajuste de descuento** — Simula variaciones de precio de venta (±50%)
- **Escenarios de competencia** — Compara 3 escenarios (actual, competencia -5%, +5%)
- **KPIs** — Unidades totales, ingresos proyectados, precio y descuento promedio
- **Gráfico diario** — Evolución de ventas con marcado de Black Friday y fines de semana
- **Tabla detallada** — Día a día con precios, descuentos e ingresos
- **Comparativa de escenarios** — Gráfico multi-línea por escenario de competencia

---

## 🛠️ Stack tecnológico

| Librería | Uso |
|---|---|
| `streamlit` | Dashboard interactivo |
| `scikit-learn` | Modelo de forecasting |
| `pandas` / `numpy` | Manipulación de datos |
| `matplotlib` / `seaborn` | Visualizaciones |
| `holidays` | Detección de festivos nacionales |
| `joblib` | Serialización del modelo |

---

## 📝 Licencia

Este proyecto es de uso educativo/práctico. Siéntete libre de adaptarlo a tus necesidades.
