"""
data_loader.py
--------------
Carga el CSV de Iris en un DataFrame con nombres de columna legibles.

El archivo original usa el formato de scikit-learn: la primera fila no es un
encabezado normal sino metadatos (150, 4, setosa, versicolor, virginica).
Por eso se salta esa fila y se asignan los nombres desde config.py.
"""
import pandas as pd

from src import config


def load_iris(path=config.RAW_DATA_PATH):
    """Devuelve un DataFrame con 4 variables + columna 'target' (0, 1, 2)."""
    df = pd.read_csv(path, skiprows=1, header=None,
                     names=config.FEATURES + [config.TARGET])
    df[config.TARGET] = df[config.TARGET].astype(int)
    return df


def split_xy(df):
    """Separa variables predictoras (X) y variable objetivo (y)."""
    return df[config.FEATURES], df[config.TARGET]
