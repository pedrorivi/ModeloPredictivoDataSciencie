import pandas as pd
import os

def verify_consistency():
    path_train = "c:/Users/Pedro/Desktop/Data_Practica/data/processed/DF.csv"
    path_infer = "c:/Users/Pedro/Desktop/Data_Practica/data/processed/inferencia_df_transformado.csv"
    
    if not os.path.exists(path_train) or not os.path.exists(path_infer):
        print("Error: No se encuentran los archivos de datos.")
        return
        
    df_train = pd.read_csv(path_train, nrows=1)
    df_infer = pd.read_csv(path_infer, nrows=1)
    
    cols_train = set(df_train.columns)
    cols_infer = set(df_infer.columns)
    
    missing_in_infer = cols_train - cols_infer
    extra_in_infer = cols_infer - cols_train
    
    print(f"Total columnas entrenamiento: {len(cols_train)}")
    print(f"Total columnas inferencia: {len(cols_infer)}")
    
    if len(missing_in_infer) > 0:
        print(f"ALERTA: Columnas faltantes en inferencia ({len(missing_in_infer)}):")
        print(sorted(list(missing_in_infer)))
    else:
        print("OK: No faltan columnas en el set de inferencia.")
        
    if len(extra_in_infer) > 0:
        print(f"NOTA: Columnas extra en inferencia ({len(extra_in_infer)}):")
        print(sorted(list(extra_in_infer)))
    else:
        print("OK: No hay columnas extra en el set de inferencia.")

if __name__ == "__main__":
    verify_consistency()
