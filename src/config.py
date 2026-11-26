"""
config.py
---------
Configuración central del pipeline. Todas las rutas y parámetros viven aquí
para que ningún otro módulo tenga valores "quemados" (hardcoded).
"""
from pathlib import Path

# Raíz del proyecto: carpeta que contiene a src/
ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Rutas de datos y artefactos ---
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "iris.csv"
CONTRACT_PATH = ROOT_DIR / "contracts" / "data_contract.json"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
MODEL_PATH = MODELS_DIR / "iris_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.json"
LOG_PATH = REPORTS_DIR / "pipeline_run.log"

# --- Nombres de columnas (el CSV original no trae encabezados descriptivos) ---
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "target"

# --- Parámetros de reproducibilidad y entrenamiento ---
RANDOM_STATE = 42      # semilla fija: mismo resultado en cada ejecución
TEST_SIZE = 0.2        # 20 % de los datos para prueba (30 registros)
CV_FOLDS = 5           # validación cruzada estratificada de 5 particiones
MAX_ITER = 1000        # iteraciones máximas de LogisticRegression

# --- Umbral de calidad: si el modelo no lo alcanza, el pipeline falla ---
MIN_TEST_ACCURACY = 0.90
