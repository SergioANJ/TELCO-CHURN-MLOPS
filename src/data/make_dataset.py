"""
make_dataset.py

Convierte el dataset crudo (raw) de Telco Customer Churn en un dataset limpio y listo para feauture engineering

Decisiones de limpieza documentadas (ver notebooks/01_eda.ipynb para el
razonamiento completo detrás de cada una):
1. customerID: eliminado (identificador, no predictivo)
2. TotalCharges: convertido a numérico; nulos (tenure=0) imputados con 0
3. Outliers: no tratados (valores legítimos de negocio)
4. Duplicados: conservados (evita agravar el desbalance de clases)

"""

import pandas as pd
from pathlib import Path

def cargar_datos_crudos(ruta_raw: str) -> pd.DataFrame:
    """
    Cargamos el dataset original sin procesar.
    Args:
        ruta_raw: ruta al archivo CSV crudo.
    Returns:
        Dataframe con los datos tal cual vienen del original
    """
    df = pd.read_csv(ruta_raw)
    return df


def limpiar_datos (df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la limpieza definida en el EDA:
    -Elimina customerID
    -Corrige el tipo de dato de TotalCharges e imputa nulos con 0
    -No trata outliers ni duplicados 

    Arg: 
        Dataframe crudo, tal como sale de cargar_datos_crudos
    
    Returns:
        DataFrame limpio.
    """
    df = df.copy()

    #1. Eliminamos identificador no predictivo
    df = df.drop(columns=['customerID'])

    #2. Corregir TotalCharges (pasar de string a númerico, nulos disfrazados)
    df['TotalCharges'] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    #Verificación de la premisa de negocio antes de imputar
    nulos = df[df['TotalCharges'].isnull()]
    assert (nulos['tenure'] == 0).all(),(
        "Se esperaba que todos los nulos de TotalCharges correspondieran"
        "a tenure=0. Revisar el dataset, la premisa de limpieza ya no aplica"
    )

    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    return df

def guardar_datos_procesados(df: pd.DataFrame, ruta_processed: str) -> None:
    """
    Guardar el dataset limpio en la carpeta de datos procesados. 

    Args:
        df: Dataframe limpio
        ruta_pruta_processed: ruta de salida (incluye nombre de archivo .csv)
    """
    Path(ruta_processed).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta_processed, index=False)
    print(f"Dataset procesado guardado en: {ruta_processed}")

def main(ruta_raw: str, ruta_processed: str)-> pd.DataFrame:
    """
    Orquesta el pipeline completo de limpieza de datos
    """
    df_crudo = cargar_datos_crudos(ruta_raw)
    df_limpio = limpiar_datos(df_crudo)
    guardar_datos_procesados(df_limpio, ruta_processed)
    return df_limpio

if __name__ =="__main__":
    RUTA_RAW = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    RUTA_PROCESSED = "data/processed/telco_churn_clean.csv"

    main(RUTA_RAW, RUTA_PROCESSED)