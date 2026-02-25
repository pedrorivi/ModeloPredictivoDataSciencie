import pandas as pd
import joblib
import os

def run_forecasting():
    print("Iniciando el proceso de prediccion (Forecasting)...")
    
    # 1. Definir rutas
    path_modelo = "models/modelo_final_forecasting.joblib"
    path_datos = "data/processed/inferencia_df_transformado.csv"
    path_output = "data/results/predicciones_noviembre_2025.csv"
    
    # 2. Cargar el modelo
    if not os.path.exists(path_modelo):
        print(f"Error: No se encontro el modelo en {path_modelo}")
        return
    
    model = joblib.load(path_modelo)
    print("Modelo cargado correctamente.")
    
    # 3. Cargar datos de inferencia
    if not os.path.exists(path_datos):
        print(f"Error: No se encontraron los datos procesados en {path_datos}")
        return
        
    df_infer = pd.read_csv(path_datos)
    print(f"Datos de inferencia cargados: {df_infer.shape[0]} registros.")
    
    # 4. Alinear características (X)
    # Extraer las columnas exactas que vio el modelo durante el entrenamiento
    if hasattr(model, 'feature_names_in_'):
        X_cols = model.feature_names_in_.tolist()
        print(f"Alineando con las {len(X_cols)} columnas del modelo.")
    else:
        # Fallback si no tiene el atributo (aunque sklearn >= 1.0 lo suele tener)
        cols_a_excluir = ['fecha', 'ingresos', 'unidades_vendidas']
        X_cols = [col for col in df_infer.columns if col not in cols_a_excluir and df_infer[col].dtype != 'object']
    
    # Asegurar que todas las columnas necesarias existen en df_infer
    for col in X_cols:
        if col not in df_infer.columns:
            df_infer[col] = 0
            
    X = df_infer[X_cols]
    
    # 5. Realizar predicciones
    print("Ejecutando predicciones...")
    df_infer['prediccion_unidades'] = model.predict(X)
    
    # Asegurar que no haya predicciones negativas
    df_infer['prediccion_unidades'] = df_infer['prediccion_unidades'].clip(lower=0)
    
    # 6. Preparar el DataFrame de resultados
    os.makedirs(os.path.dirname(path_output), exist_ok=True)
    
    # Para el reporte, queremos conservar columnas clave
    # Pero df_infer ya tiene todo. Guardamos lo esencial + prediccion
    # Si 'nombre' original fue borrado por OHE, buscaremos el producto_id
    output_cols = ['fecha', 'producto_id']
    if 'nombre' in df_infer.columns: output_cols.append('nombre')
    output_cols.append('prediccion_unidades')
    
    df_final = df_infer.copy()
    
    df_infer.to_csv(path_output, index=False)
    print(f"Predicciones guardadas en: {path_output}")
    
    # 7. Resumen estadístico
    print("\n--- RESUMEN DE FORECASTING (NOVIEMBRE 2025) ---")
    total_ventas = df_infer['prediccion_unidades'].sum()
    print(f"Ventas Totales Proyectadas: {total_ventas:.2f} unidades")
    
    if 'producto_id' in df_infer.columns:
        top_ventas = df_infer.groupby('producto_id')['prediccion_unidades'].sum().sort_values(ascending=False).head(5)
        print("\nTop 5 Productos con más ventas proyectadas:")
        print(top_ventas)

if __name__ == "__main__":
    run_forecasting()
