import pandas as pd
import numpy as np
import holidays
import os

def prepare_inference():
    print("Iniciando preparacion de datos de inferencia...")
    
    # 1. Carga de datos
    path_raw = "../data/raw/inferencia/ventas_2025_inferencia.csv"
    if not os.path.exists(path_raw):
        # Intentar ruta absoluta si la relativa falla
        path_raw = "c:/Users/Pedro/Desktop/Data_Practica/data/raw/inferencia/ventas_2025_inferencia.csv"
        
    df = pd.read_csv(path_raw)
    
    # 2. Convertir fecha y filtrar por Noviembre 2025 (Inferencia)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df[df['fecha'].dt.month == 11]
    
    # 3. Variables Temporales
    df['mes'] = df['fecha'].dt.month
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['es_fin_semana'] = df['dia_semana'].isin([5, 6])
    df['dia'] = df['fecha'].dt.day
    df['anio'] = df['fecha'].dt.year
    
    es_holidays = holidays.Spain()
    df['es_festivo'] = df['fecha'].apply(lambda x: x in es_holidays)
    
    def get_estacion(mes):
        if mes in [12, 1, 2]: return 'Invierno'
        if mes in [3, 4, 5]: return 'Primavera'
        if mes in [6, 7, 8]: return 'Verano'
        return 'Otono'
    
    df['estacion'] = df['mes'].apply(get_estacion)
    
    # 4. Métricas de Competencia
    df['precio_min_competencia'] = df[['Amazon', 'Decathlon', 'Deporvillage']].min(axis=1)
    df['gap_precio'] = df['precio_venta'] - df['precio_min_competencia']
    df['gap_porcentaje'] = (df['gap_precio'] / df['precio_min_competencia']) * 100
    df['promo_rival'] = df['gap_porcentaje'] > 10
    
    df['prec_competencia'] = df[['Amazon', 'Decathlon', 'Deporvillage']].min(axis=1)
    df['ratio_precio'] = df['precio_venta'] / df['prec_competencia']
    df['ratio_competencia'] = df['precio_venta'] / df['precio_min_competencia']
    
    # 5. Margen
    df['margen_unitario'] = df['precio_venta'] - df['precio_base']
    df['margen_total'] = df['margen_unitario'] * df['unidades_vendidas']
    
    # 6. Eventos especiales (Noviembre 2025)
    df['es_black_friday'] = False
    df.loc[(df['fecha'].dt.month == 11) & (df['fecha'].dt.day == 28), 'es_black_friday'] = True
    df['es_cyber_monday'] = False
    df['es_quincena'] = df['fecha'].dt.day.isin([15, 30])
    df['es_cobro'] = df['fecha'].dt.day.isin([28, 29, 30, 31, 1, 2, 3])
    
    # 7. Lags y Medias Móviles (Inicializados en 0 para este bloque de inferencia)
    lag_cols = ['ventas_lag1', 'ventas_lag7', 'unidades_vendidas_lag1', 
                'unidades_vendidas_lag2', 'unidades_vendidas_lag3', 
                'unidades_vendidas_lag4', 'unidades_vendidas_lag5', 
                'unidades_vendidas_lag6', 'unidades_vendidas_lag7', 'media_movil_7']
    for col in lag_cols:
        df[col] = 0
        
    # 8. Descuento
    df['descuento_porcentaje'] = ((df['precio_base'] - df['precio_venta']) / df['precio_base']) * 100
    
    # 9. One-Hot Encoding
    cols_categorical = ['nombre', 'categoria', 'subcategoria']
    # Crear dummies pero mantener columnas originales
    df_dummies = pd.get_dummies(df[cols_categorical], prefix=[c + '_h' for c in cols_categorical])
    df = pd.concat([df, df_dummies], axis=1)
    
    # 10. Alineación de columnas con el set de entrenamiento
    path_train = "c:/Users/Pedro/Desktop/Data_Practica/data/processed/DF.csv"
    if os.path.exists(path_train):
        df_train_cols = pd.read_csv(path_train, nrows=0).columns.tolist()
        
        # Filtrar columnas que no están en entrenamiento (como Amazon, Decathlon, Deporvillage)
        # y asegurar que todas las de entrenamiento existen (rellenando con 0 si faltan dummies)
        for col in df_train_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Reordenar y filtrar para que coincida exactamente con entrenamiento
        df = df[df_train_cols]
        print("Alineacion de columnas completada con exito.")
    else:
        print("Advertencia: No se encontro el archivo de entrenamiento para alinear columnas.")
    
    # Guardar resultado
    output_path = "data/processed/inferencia_df_transformado.csv"
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Datos de inferencia preparados y guardados en: {output_path}")

if __name__ == "__main__":
    prepare_inference()
