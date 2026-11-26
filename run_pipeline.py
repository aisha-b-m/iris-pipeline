"""
run_pipeline.py
===============
Pipeline reproducible del dataset Iris en 4 etapas, ejecutable con UN comando:

    python run_pipeline.py               # ejecuta las 4 etapas
    python run_pipeline.py --skip-tests  # ejecuta solo las etapas 1 a 3

Etapas:
    1. Preparación de datos      -> carga, tipado y partición 80 % / 20 %
    2. Validación y calidad      -> tipos, nulos, rangos, tamaños y fugas
    3. Entrenamiento + MLflow    -> modelo, métricas (accuracy y F1) y registro
    4. Pruebas automatizadas     -> pytest sobre todo el proyecto

Si una etapa falla, el pipeline se detiene y termina con código 1.
Toda la salida queda en reports/pipeline_run.log como evidencia.

Para ver los experimentos registrados en MLflow:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
y abrir http://127.0.0.1:5000 en el navegador.
"""

# ---------------------------------------------------------------------------
# IMPORTACIONES
# ---------------------------------------------------------------------------
import argparse                     # lee argumentos de la terminal (--skip-tests)
import json                         # escribe las métricas en formato JSON
import os                           # cambia el directorio de trabajo
import platform                     # obtiene versión de Python y sistema operativo
import subprocess                   # ejecuta pytest como un proceso aparte
import sys                          # acceso al intérprete actual y a sys.exit
import time                         # mide cuánto tarda el pipeline
from datetime import datetime       # fecha y hora para el log
from pathlib import Path            # manejo de rutas que funciona en Windows y Linux

ROOT = Path(__file__).resolve().parent   # carpeta donde está este archivo (raíz del repo)
os.chdir(ROOT)                           # todo se ejecuta desde la raíz, sin importar desde dónde se llame
sys.path.insert(0, str(ROOT))            # permite importar el paquete "src"

import joblib                                           # guarda el modelo entrenado en disco
import pandas as pd                                     # manipulación de tablas
from sklearn.linear_model import LogisticRegression     # algoritmo de clasificación
from sklearn.metrics import accuracy_score, f1_score    # métricas pedidas para la etapa 3
from sklearn.model_selection import train_test_split    # partición 80 / 20
from sklearn.pipeline import Pipeline                   # encadena escalado + modelo
from sklearn.preprocessing import StandardScaler        # estandariza las variables

from src import config                                  # rutas y parámetros centralizados
from src.validate import load_contract, validate_record, validate_training_file  # validación sin librerías externas

# ---------------------------------------------------------------------------
# PARÁMETROS DEL PIPELINE
# ---------------------------------------------------------------------------
PROCESSED_DIR = ROOT / "data" / "processed"     # carpeta para los datos ya preparados
TRAIN_PATH = PROCESSED_DIR / "train.csv"        # 80 % de los registros
TEST_PATH = PROCESSED_DIR / "test.csv"          # 20 % de los registros
TEST_SIZE = 0.20                                # proporción de prueba (20 %)
MLFLOW_URI = "sqlite:///mlflow.db"              # base de datos local donde MLflow guarda los experimentos
EXPERIMENT = "iris-pipeline"                    # nombre del experimento en MLflow
LOG_PATH = ROOT / "reports" / "pipeline_run.log"  # archivo de evidencia

_log_file = None   # se abre en main(); log() escribe en pantalla y en este archivo


def log(msg=""):
    """Imprime un mensaje en pantalla y lo guarda en el archivo de log."""
    print(msg)                          # muestra en la terminal
    if _log_file:                       # si el archivo de log está abierto...
        _log_file.write(msg + "\n")     # ...escribe la misma línea
        _log_file.flush()               # fuerza la escritura inmediata (por si el proceso se cae)


class StageError(Exception):
    """Error que detiene el pipeline cuando una etapa no cumple sus condiciones."""


# ===========================================================================
# ETAPA 1: PREPARACIÓN DE DATOS
# ===========================================================================
def stage_1_prepare():
    """Carga el CSV crudo, asigna nombres y tipos, y lo divide 80 % / 20 %."""
    # El CSV crudo trae en la 1.ª fila metadatos (150,4,setosa,...), no nombres de
    # columna; por eso se salta con skiprows=1 y los nombres se ponen a mano.
    df = pd.read_csv(config.RAW_DATA_PATH, skiprows=1, header=None,
                     names=config.FEATURES + [config.TARGET])
    log(f"    registros cargados: {len(df)} | columnas: {list(df.columns)}")

    # Se fuerzan los tipos esperados: medidas como decimales y especie como entero.
    # errors="coerce" convierte lo que no sea número en NaN para que la etapa 2 lo detecte.
    for col in config.FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df[config.TARGET] = pd.to_numeric(df[config.TARGET], errors="coerce").astype("Int64")

    # Se guarda el número de fila original como identificador. Sirve en la
    # etapa 2 para comprobar que ningún registro quedó en train y test a la vez.
    df.insert(0, "row_id", range(len(df)))

    # Los duplicados exactos no se eliminan (Iris tiene uno legítimo), solo se informan.
    n_dup = int(df.duplicated(subset=config.FEATURES + [config.TARGET]).sum())
    log(f"    filas duplicadas (informativo): {n_dup}")

    # Partición 80 / 20:
    #  - test_size=0.20      -> 20 % para prueba (30 registros) y 80 % para entrenar (120)
    #  - stratify=target     -> cada parte conserva 1/3 de cada especie
    #  - random_state fijo   -> la partición es idéntica en cada ejecución (reproducible)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[config.TARGET],
        random_state=config.RANDOM_STATE,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)   # crea data/processed si no existe
    train_df.to_csv(TRAIN_PATH, index=False)           # guarda el 80 %
    test_df.to_csv(TEST_PATH, index=False)             # guarda el 20 %

    log(f"    entrenamiento: {len(train_df)} registros ({len(train_df) / len(df):.0%})")
    log(f"    prueba:        {len(test_df)} registros ({len(test_df) / len(df):.0%})")
    log(f"    archivos: {TRAIN_PATH.relative_to(ROOT)}, {TEST_PATH.relative_to(ROOT)}")
    return train_df, test_df                           # se pasan a la etapa siguiente


# ===========================================================================
# ETAPA 2: VALIDACIÓN Y CONTROL DE CALIDAD
# ===========================================================================
def stage_2_validate(train_df, test_df):
    """
    Verifica los datos antes de entrenar. Usa src/validate.py, que NO depende
    de librerías externas y toma las reglas del Contrato de Datos.
    """
    contract = load_contract()   # reglas de tipos y rangos desde contracts/data_contract.json
    problems = []                # aquí se acumulan todos los errores encontrados

    # 2.1 Archivo crudo: tipos, nulos, rangos, estructura y clases permitidas
    report = validate_training_file(config.RAW_DATA_PATH, contract)
    log(f"    [archivo crudo] {report['status']}: "
        f"{report['valid_rows']}/{report['total_rows']} filas válidas")
    problems += [f"crudo -> {e}" for e in report["errors"]]

    # 2.2 Cada registro de train y test se valida contra el contrato (tipos, nulos, rangos)
    for name, part in (("train", train_df), ("test", test_df)):
        # Se convierte cada fila a diccionario de floats, como llegaría en inferencia.
        # Los NaN se cambian por None para que el validador los reporte como nulos.
        records = part[config.FEATURES].astype(object).where(part[config.FEATURES].notna(), None)
        n_bad = 0
        for row_id, rec in zip(part["row_id"], records.to_dict("records")):
            errs = validate_record(rec, contract)       # lista vacía = registro válido
            if errs:
                n_bad += 1
                problems += [f"{name} fila {row_id} -> {e}" for e in errs]
        # La etiqueta también debe existir y ser una de las clases permitidas (0, 1, 2)
        allowed = contract["training_schema"]["target_allowed_values"]
        bad_target = ~part[config.TARGET].isin(allowed) | part[config.TARGET].isna()
        problems += [f"{name} fila {r} -> target inválido" for r in part.loc[bad_target, "row_id"]]
        log(f"    [{name}] {len(part) - n_bad}/{len(part)} registros cumplen el contrato")

    # 2.3 Control de calidad de la partición
    total = len(train_df) + len(test_df)
    test_share = len(test_df) / total
    if abs(test_share - TEST_SIZE) > 0.01:                  # tolerancia de 1 punto porcentual
        problems.append(f"proporción de prueba {test_share:.2%}, se esperaba {TEST_SIZE:.0%}")
    log(f"    [partición] proporción train/test: {1 - test_share:.0%} / {test_share:.0%}")

    overlap = set(train_df["row_id"]) & set(test_df["row_id"])   # registros repetidos entre partes
    if overlap:
        problems.append(f"fuga de datos: {len(overlap)} registros están en train y en test")
    log(f"    [partición] registros compartidos entre train y test: {len(overlap)}")

    # Cada especie debe tener la misma proporción en train y en test (estratificación)
    p_train = train_df[config.TARGET].value_counts(normalize=True).sort_index()
    p_test = test_df[config.TARGET].value_counts(normalize=True).sort_index()
    max_gap = (p_train - p_test).abs().max()
    if max_gap > 0.05:                                       # máx. 5 puntos de diferencia
        problems.append(f"clases desbalanceadas entre train y test (diferencia {max_gap:.2%})")
    conteo = test_df[config.TARGET].value_counts().sort_index()          # registros por especie en test
    log(f"    [partición] clases en test: {dict(zip(conteo.index.astype(int), conteo.astype(int)))}")

    # Si hubo cualquier problema, se muestran (máx. 20) y se detiene el pipeline
    if problems:
        for p in problems[:20]:
            log(f"    ERROR: {p}")
        raise StageError(f"{len(problems)} problemas de calidad de datos")
    log("    control de calidad superado: sin nulos, tipos correctos, rangos válidos, sin fugas")


# ===========================================================================
# ETAPA 3: ENTRENAMIENTO Y REGISTRO EN MLFLOW
# ===========================================================================
def stage_3_train(train_df, test_df):
    """Entrena el modelo, calcula accuracy y F1, lo guarda y lo registra en MLflow."""
    try:
        import mlflow                # se importa aquí para que las etapas 1 y 2 funcionen sin MLflow
        import mlflow.sklearn        # módulo de MLflow para modelos de scikit-learn
        from mlflow.models import infer_signature   # describe entradas/salidas del modelo
    except ImportError as exc:       # mensaje claro si falta instalar MLflow
        raise StageError("MLflow no está instalado. Ejecute: pip install -r requirements.txt") from exc

    # Separación de variables predictoras (X) y variable objetivo (y)
    X_train, y_train = train_df[config.FEATURES], train_df[config.TARGET].astype(int)
    X_test, y_test = test_df[config.FEATURES], test_df[config.TARGET].astype(int)

    # Hiperparámetros: se definen en un diccionario para registrarlos en MLflow tal cual
    params = {"max_iter": config.MAX_ITER, "random_state": config.RANDOM_STATE}

    # Pipeline de sklearn: el escalador aprende media y desviación SOLO con train
    # y aplica esas mismas cifras a test; así se evita la fuga de información.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**params)),
    ])
    model.fit(X_train, y_train)          # entrenamiento con el 80 %

    y_pred = model.predict(X_test)       # predicciones sobre el 20 % nunca visto
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),                  # % de aciertos
        "f1_score": f1_score(y_test, y_pred, average="macro"),       # F1 promedio de las 3 clases
    }
    log(f"    accuracy (test): {metrics['accuracy']:.4f}")
    log(f"    F1 macro (test): {metrics['f1_score']:.4f}")

    # Compuerta de calidad: si el modelo es peor que el umbral, no se aprueba
    if metrics["accuracy"] < config.MIN_TEST_ACCURACY:
        raise StageError(f"accuracy {metrics['accuracy']:.4f} < umbral {config.MIN_TEST_ACCURACY}")

    # Almacenamiento local del modelo (lo usan predict.py y las pruebas)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "X_test": X_test, "y_test": y_test,
                 "features": config.FEATURES}, config.MODEL_PATH)
    log(f"    modelo guardado en {config.MODEL_PATH.relative_to(ROOT)}")

    # Métricas en JSON (evidencia legible sin abrir MLflow)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(
        {**{k: round(float(v), 4) for k, v in metrics.items()},
         "n_train": len(X_train), "n_test": len(X_test), **params}, indent=2))

    # ---------------- Registro en MLflow ----------------
    mlflow.set_tracking_uri(MLFLOW_URI)        # dónde se guardan los experimentos (mlflow.db)
    mlflow.set_experiment(EXPERIMENT)          # crea el experimento si no existe
    with mlflow.start_run(run_name=f"logreg-{datetime.now():%Y%m%d-%H%M%S}") as run:
        mlflow.log_params(params)                                   # hiperparámetros
        mlflow.log_params({"test_size": TEST_SIZE,                  # cómo se partieron los datos
                           "n_train": len(X_train), "n_test": len(X_test)})
        mlflow.log_metrics(metrics)                                 # accuracy y f1_score
        signature = infer_signature(X_train, model.predict(X_train))  # esquema de entrada/salida
        mlflow.sklearn.log_model(model, name="model",               # modelo como artefacto
                                 signature=signature, input_example=X_train.head(3))
        mlflow.log_artifact(str(config.CONTRACT_PATH))              # contrato de datos usado
        log(f"    MLflow: experimento '{EXPERIMENT}', run_id {run.info.run_id}")
    return metrics


# ===========================================================================
# ETAPA 4: PRUEBAS AUTOMATIZADAS
# ===========================================================================
def stage_4_tests():
    """Ejecuta pytest en un proceso aparte y falla si alguna prueba no pasa."""
    result = subprocess.run(                         # lanza "python -m pytest tests -v"
        [sys.executable, "-m", "pytest", "tests", "-v"],
        cwd=ROOT, capture_output=True, text=True,    # captura la salida para el log
    )
    for line in result.stdout.rstrip().splitlines():
        log(f"    {line}")                           # copia la salida de pytest al log
    if result.returncode != 0:                       # código distinto de 0 = alguna prueba falló
        log(result.stderr)
        raise StageError("una o más pruebas automatizadas fallaron")


# ===========================================================================
# ORQUESTACIÓN
# ===========================================================================
def main():
    global _log_file
    parser = argparse.ArgumentParser(description="Pipeline reproducible Iris (4 etapas)")
    parser.add_argument("--skip-tests", action="store_true", help="omite la etapa 4")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)    # crea reports/ si no existe
    _log_file = open(LOG_PATH, "w", encoding="utf-8")     # abre el log (se sobrescribe en cada ejecución)
    start = time.time()                                   # marca de tiempo inicial

    log("=" * 64)
    log(f"PIPELINE IRIS | {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Python {platform.python_version()} | {platform.system()} {platform.release()}")
    log("=" * 64)

    try:
        log("\n>>> ETAPA 1: Preparación de datos")
        train_df, test_df = stage_1_prepare()
        log("[OK] Etapa 1")

        log("\n>>> ETAPA 2: Validación y control de calidad")
        stage_2_validate(train_df, test_df)
        log("[OK] Etapa 2")

        log("\n>>> ETAPA 3: Entrenamiento y registro en MLflow")
        stage_3_train(train_df, test_df)
        log("[OK] Etapa 3")

        if args.skip_tests:
            log("\n>>> ETAPA 4: omitida (--skip-tests)")
        else:
            log("\n>>> ETAPA 4: Pruebas automatizadas")
            stage_4_tests()
            log("[OK] Etapa 4")
    except Exception as exc:                              # cualquier fallo detiene el pipeline
        log(f"\n[X] PIPELINE DETENIDO: {type(exc).__name__}: {exc}")
        _log_file.close()
        return 1                                          # código 1 = fallo (lo detectan make y GitHub Actions)

    log(f"\nPIPELINE COMPLETADO en {time.time() - start:.1f} s")
    log(f"Evidencia: {LOG_PATH.relative_to(ROOT)} | Experimentos: mlflow ui --backend-store-uri {MLFLOW_URI}")
    _log_file.close()
    return 0                                              # código 0 = éxito


if __name__ == "__main__":        # solo se ejecuta si se llama directamente, no al importar
    sys.exit(main())
