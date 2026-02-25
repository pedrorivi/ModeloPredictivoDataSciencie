# 📚 Guía de Estudio: Fundamentos de Data Science
### Aprendiendo con el proyecto de Forecasting de Ventas

> Esta guía recorre todos los pasos del proyecto **paso a paso**, explicando el *qué*, el *por qué* y el *cómo* de cada decisión. Al terminar, comprenderás el flujo completo de un proyecto de Machine Learning real.

---

## 🗺️ Mapa del proceso

```
1. Entender el problema
        ↓
2. Recopilar y explorar los datos (EDA)
        ↓
3. Limpiar los datos
        ↓
4. Ingeniería de características (Feature Engineering)
        ↓
5. Preparar el modelo (Train / Test split)
        ↓
6. Entrenar el modelo
        ↓
7. Evaluar el modelo
        ↓
8. Preparar datos de inferencia
        ↓
9. Predecir (Inferencia)
        ↓
10. Visualizar y comunicar resultados
```

---

## 1. 🎯 Entender el problema

Antes de tocar ningún dato, el Data Scientist debe responder:

| Pregunta | Respuesta en este proyecto |
|---|---|
| ¿Qué queremos predecir? | Unidades vendidas por día |
| ¿Para qué período? | Noviembre 2025 |
| ¿Qué tipo de problema es? | **Regresión** (predecimos un número continuo) |
| ¿Cómo mediremos el éxito? | MAE, RMSE, comparar con baseline |

> 💡 **Concepto clave — Regresión vs Clasificación:**
> - **Regresión**: la salida es un número (e.g. 42 unidades, 150€).
> - **Clasificación**: la salida es una categoría (e.g. "venderá mucho" / "venderá poco").

---

## 2. 🔍 Exploración de Datos (EDA)

**EDA = Exploratory Data Analysis.** Es la fase donde "conocemos" nuestros datos.

### DataFrames del proyecto

```python
import pandas as pd

# DataFrame principal de ventas históricas
df = pd.read_csv("ventas.csv")

# DataFrame de precios de competencia
df_competencia = pd.read_csv("competencia.csv")
# Columnas: ['fecha', 'producto_id', 'Amazon', 'Decathlon', 'Deporvillage']
```

### Comandos esenciales de EDA

```python
# Ver las primeras filas
df.head()

# Resumen estadístico: media, std, min, max, cuartiles
df.describe()

# Tipos de datos y valores nulos
df.info()
df.isnull().sum()

# Distribución de una columna
df['unidades_vendidas'].hist(bins=30)

# Correlaciones entre variables numéricas
df.corr()['unidades_vendidas'].sort_values(ascending=False)
```

> 💡 **Correlación**: mide la relación lineal entre dos variables. Va de -1 (relación inversa perfecta) a +1 (relación directa perfecta). 0 = sin relación lineal.

---

## 3. 🧹 Limpieza de Datos

Los datos reales siempre tienen problemas. Los más comunes:

| Problema | Solución habitual |
|---|---|
| Valores nulos (NaN) | Rellenar con media, mediana, 0 o eliminar |
| Duplicados | `df.drop_duplicates()` |
| Tipos incorrectos | `pd.to_datetime()`, `astype()` |
| Outliers | Detectar con IQR o Z-score, decidir si eliminar |

```python
# Convertir fecha a tipo datetime (imprescindible para series temporales)
df['fecha'] = pd.to_datetime(df['fecha'])

# Rellenar nulos con la mediana (más robusto que la media ante outliers)
df['precio_venta'].fillna(df['precio_venta'].median(), inplace=True)

# Eliminar duplicados
df.drop_duplicates(inplace=True)
```

---

## 4. ⚙️ Ingeniería de Características (Feature Engineering)

Esta es la fase más creativa y **la que más impacto tiene** en el rendimiento del modelo.

> 💡 **Feature**: cada columna que usa el modelo para aprender. Un buen feature es información relevante que ayuda al modelo a predecir.

### 4.1 Variables Temporales

```python
# Extraer información de la fecha
df['mes']        = df['fecha'].dt.month        # 1-12
df['dia']        = df['fecha'].dt.day          # 1-31
df['dia_semana'] = df['fecha'].dt.dayofweek    # 0=Lun, 6=Dom
df['anio']       = df['fecha'].dt.year

# Variable binaria: ¿es fin de semana?
df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
```

### 4.2 Festivos

```python
import holidays

festivos_es = holidays.Spain(years=[2024, 2025])

df['es_festivo'] = df['fecha'].apply(lambda x: 1 if x in festivos_es else 0)
```

### 4.3 Eventos Especiales (Black Friday, etc.)

```python
# Black Friday = 4º viernes de noviembre
def es_black_friday(fecha):
    if fecha.month != 11:
        return 0
    viernes = [d for d in pd.date_range(f"{fecha.year}-11-01", f"{fecha.year}-11-30")
               if d.weekday() == 4]
    return 1 if fecha == viernes[3] else 0  # el 4º viernes

df['es_black_friday'] = df['fecha'].apply(es_black_friday)
```

### 4.4 Variables de Lag (Retardos)

Los **lags** son valores pasados de la variable objetivo. Son críticos en series temporales porque las ventas de hoy dependen de las de ayer.

```python
# Lag 1: ventas del día anterior
df['unidades_vendidas_lag1'] = df.groupby('producto_id')['unidades_vendidas'].shift(1)

# Lag 7: ventas de hace una semana (mismo día de la semana)
df['unidades_vendidas_lag7'] = df.groupby('producto_id')['unidades_vendidas'].shift(7)
```

> ⚠️ **¡Cuidado con el data leakage!** Los lags deben calcularse con `shift()` para que el modelo NUNCA vea datos del futuro durante el entrenamiento.

### 4.5 Media Móvil

Suaviza las fluctuaciones y captura la tendencia reciente:

```python
# Media de los últimos 7 días
df['media_movil_7'] = (
    df.groupby('producto_id')['unidades_vendidas']
    .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
)
```

### 4.6 Variables de Competencia y Precio

```python
# Gap de precio respecto a la competencia
df['gap_precio']      = df['precio_venta'] - df['precio_min_competencia']
df['gap_porcentaje']  = (df['gap_precio'] / df['precio_min_competencia']) * 100

# Ratio: cuánto más caro somos que la competencia
df['ratio_competencia'] = df['precio_venta'] / df['precio_min_competencia']

# Descuento aplicado sobre precio base
df['descuento_porcentaje'] = ((df['precio_base'] - df['precio_venta']) / df['precio_base']) * 100
```

### 4.7 One-Hot Encoding (OHE)

Los modelos de ML no entienden texto. Hay que convertir categorías a números:

```python
# Sin OHE: columna 'nombre' = "Nike Air Zoom Pegasus 40"
# Con OHE: columna 'nombre_h_Nike Air Zoom Pegasus 40' = 1, el resto = 0

df = pd.get_dummies(df, columns=['nombre', 'categoria', 'subcategoria'], prefix_sep='_h_')
```

> 💡 El sufijo `_h_` es una convención propia del proyecto para identificar las columnas codificadas.

---

## 5. ✂️ División Train / Test

Separamos los datos para poder **evaluar el modelo en datos que nunca ha visto**.

```python
from sklearn.model_selection import train_test_split

# En series temporales: NO usar train_test_split aleatorio.
# Siempre dividir por tiempo: los datos más antiguos para train, los más recientes para test.

fecha_corte = pd.Timestamp("2025-09-30")

df_train = df[df['fecha'] <= fecha_corte]
df_test  = df[df['fecha'] >  fecha_corte]

FEATURES = [col for col in df.columns if col not in ['fecha', 'unidades_vendidas', 'ingresos']]
TARGET   = 'unidades_vendidas'

X_train, y_train = df_train[FEATURES], df_train[TARGET]
X_test,  y_test  = df_test[FEATURES],  df_test[TARGET]
```

> ⚠️ **Data Leakage**: si usamos datos del futuro para entrenar, el modelo "hace trampa". El modelo aprenderá patrones que no existirán en producción y sus métricas serán falsamente buenas.

---

## 6. 🤖 Entrenar el Modelo

### ¿Por qué HistGradientBoostingRegressor?

Es un modelo de la familia **Gradient Boosting**: construye muchos árboles de decisión pequeños en secuencia, donde cada árbol corrige los errores del anterior.

| Ventaja | Detalle |
|---|---|
| Robusto ante outliers | Usa árboles, no afectados por escala |
| Maneja NaNs nativamente | No necesita imputación previa |
| Muy preciso | Estado del arte en datos tabulares |
| Rápido | Versión histograma, mucho más veloz que GBR clásico |

```python
from sklearn.ensemble import HistGradientBoostingRegressor

modelo = HistGradientBoostingRegressor(
    max_iter=500,          # Número de árboles
    max_depth=6,           # Profundidad máxima de cada árbol
    learning_rate=0.05,    # Paso de aprendizaje (más bajo = más lento pero más preciso)
    min_samples_leaf=20,   # Regularización: mínimo de muestras en hoja
    l2_regularization=0.1, # Penalización L2 para evitar sobreajuste
    random_state=42
)

modelo.fit(X_train, y_train)
```

> 💡 **Hiperparámetros**: son los "botones" del modelo que tú configuras antes de entrenar. El modelo aprende los **parámetros** (pesos de los árboles) solo, pero los hiperparámetros los defines tú.

---

## 7. 📊 Evaluar el Modelo

### Métricas de regresión

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

y_pred = modelo.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE:  {mae:.2f} unidades")   # Error medio absoluto
print(f"RMSE: {rmse:.2f} unidades")  # Penaliza más los errores grandes
```

| Métrica | Fórmula | Interpretación |
|---|---|---|
| **MAE** | `mean(|real - pred|)` | "Me equivoco en promedio X unidades" |
| **RMSE** | `sqrt(mean((real-pred)²))` | Más sensible a errores grandes |
| **R²** | varianza explicada | 1.0 = perfecto, 0 = igual que la media |

### Comparar con Baseline

El **baseline** es el modelo más simple posible. Si nuestro modelo no lo supera, no aporta valor.

```python
# Baseline ingenuo: predecir siempre la media histórica
baseline_pred = np.full(len(y_test), y_train.mean())
baseline_mae  = mean_absolute_error(y_test, baseline_pred)

print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Modelo MAE:   {mae:.2f}")
print(f"Mejora:       {((baseline_mae - mae) / baseline_mae * 100):.1f}%")
```

### Guardar el modelo entrenado

```python
import joblib

joblib.dump(modelo, "models/modelo_final_forecasting.joblib")

# Para cargarlo después:
modelo_cargado = joblib.load("models/modelo_final_forecasting.joblib")
```

---

## 8. 🔄 Preparar Datos de Inferencia

**Inferencia** = usar el modelo entrenado para predecir datos nuevos (el futuro).

La regla de oro: **los datos de inferencia deben tener exactamente las mismas columnas y transformaciones que los de entrenamiento**.

```python
# El modelo recuerda qué features usó durante el entrenamiento
FEATURES_MODELO = modelo.feature_names_in_.tolist()

# Verificar que los datos de inferencia tienen las mismas columnas
cols_train = set(FEATURES_MODELO)
cols_infer = set(df_inferencia.columns)

faltan_en_infer = cols_train - cols_infer
sobran_en_infer = cols_infer - cols_train

print(f"Faltan: {faltan_en_infer}")  # Deben estar todas → set()
print(f"Sobran: {sobran_en_infer}")  # Pueden ignorarse
```

> 💡 Este es el error más común en producción: el modelo fue entrenado con unas columnas y en inferencia llegan datos con columnas distintas o en distinto orden.

---

## 9. 🔮 Predicción Recursiva

En series temporales, predecimos el futuro **día a día**, usando las predicciones anteriores como lags.

```python
predicciones = []

for i, row in df_noviembre.iterrows():
    # Actualizar lag1 con la predicción del día anterior
    if len(predicciones) > 0:
        row['unidades_vendidas_lag1'] = predicciones[-1]

    # Actualizar media móvil con las últimas predicciones
    if len(predicciones) >= 7:
        row['media_movil_7'] = np.mean(predicciones[-7:])

    # Predecir
    X = pd.DataFrame([row])[FEATURES_MODELO].fillna(0)
    pred = float(modelo.predict(X)[0])
    pred = max(0, pred)  # Las ventas no pueden ser negativas

    predicciones.append(pred)
```

> ⚠️ **Error de acumulación**: en predicción recursiva, el error se acumula. Cuanto más lejos en el futuro, mayor la incertidumbre. Por eso se suelen dar intervalos de confianza.

---

## 10. 📈 Visualizar y Comunicar Resultados

Un modelo por sí solo no vale nada si no se comunican los resultados. En este proyecto, usamos **Streamlit** para crear un dashboard interactivo.

```python
import streamlit as st
import matplotlib.pyplot as plt

# KPIs en el dashboard
st.metric("📦 Unidades Totales", f"{total_unidades:,.0f} u")
st.metric("💶 Ingresos Proyectados", f"€ {total_ingresos:,.2f}")

# Gráfico de predicción diaria
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(resultados['dia'], resultados['unidades'], color="#a78bfa", linewidth=2.5)
ax.set_xlabel("Día de Noviembre")
ax.set_ylabel("Unidades Proyectadas")
st.pyplot(fig)
```

---

## 🧩 Conceptos Clave Resumidos

| Concepto | Definición simple |
|---|---|
| **Feature** | Variable de entrada del modelo |
| **Target** | Variable que queremos predecir |
| **Lag** | Valor pasado de una variable |
| **Overfitting** | El modelo memoriza el train pero falla en datos nuevos |
| **Underfitting** | El modelo es demasiado simple para capturar patrones |
| **Train/Test split** | Separar datos para entrenar y evaluar |
| **Pipeline** | Secuencia reproducible de transformaciones |
| **Hiperparámetro** | Configuración del modelo que defines tú |
| **Baseline** | Modelo mínimo de referencia para comparar |
| **Inferencia** | Predecir con el modelo ya entrenado |
| **Data Leakage** | Usar información del futuro al entrenar (trampa) |
| **OHE** | Convertir categorías a columnas binarias 0/1 |
| **MAE** | Error medio absoluto (unidades de la predicción) |
| **RMSE** | Error cuadrático medio (penaliza más los grandes errores) |

---

## 📖 Recursos para seguir aprendiendo

- **Libros**: *Hands-On Machine Learning* (Aurélien Géron) — el más completo
- **Cursos**: Kaggle Learn (gratuito, muy práctico) → [kaggle.com/learn](https://kaggle.com/learn)
- **Documentación**: [scikit-learn.org](https://scikit-learn.org) — con ejemplos por cada algoritmo
- **Series temporales**: busca "Time Series Forecasting scikit-learn" o prueba la librería `statsmodels`

---

*Guía generada a partir del proyecto Data_Practica — Forecasting de Ventas Deportivas*
