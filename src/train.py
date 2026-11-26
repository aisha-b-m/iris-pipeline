"""
train.py
--------
Entrena el modelo de clasificación y guarda el artefacto en models/.

Pasos:
    1. Valida el archivo de entrada (si los datos fallan, no se entrena).
    2. Divide en entrenamiento (80 %) y prueba (20 %) de forma estratificada.
    3. Construye un Pipeline de sklearn: StandardScaler + LogisticRegression.
    4. Estima el desempeño con validación cruzada (5 particiones) sobre train.
    5. Ajusta el modelo final y lo guarda junto con el conjunto de prueba
       (para que evaluate.py use exactamente los mismos datos).

Uso:
    python -m src.train
"""
import json
import sys

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_loader import load_iris, split_xy
from src.validate import validate_training_file


def build_model():
    """
    Pipeline de sklearn. Encadenar el escalado con el modelo evita
    fuga de información (data leakage): el scaler se ajusta solo con train.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=config.MAX_ITER,
                                   random_state=config.RANDOM_STATE)),
    ])


def main():
    # 1) Compuerta de calidad de datos
    report = validate_training_file(config.RAW_DATA_PATH)
    if report["status"] != "OK":
        print("[train] los datos no pasaron la validación; se cancela el entrenamiento.")
        return 1

    # 2) Carga y partición estratificada (mantiene 1/3 de cada especie)
    df = load_iris()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE,
    )
    print(f"[train] train: {len(X_train)} filas | test: {len(X_test)} filas")

    # 3-4) Validación cruzada sobre el conjunto de entrenamiento
    model = build_model()
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"[train] accuracy CV ({config.CV_FOLDS} folds): "
          f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 5) Ajuste final y guardado del artefacto
    model.fit(X_train, y_train)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "X_test": X_test, "y_test": y_test,
         "features": config.FEATURES},
        config.MODEL_PATH,
    )

    # Métricas de entrenamiento (evaluate.py agrega las de prueba)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_metrics = {
        "cv_folds": config.CV_FOLDS,
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "cv_scores": [round(float(s), 4) for s in cv_scores],
        "train_accuracy": round(float(model.score(X_train, y_train)), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "random_state": config.RANDOM_STATE,
    }
    with open(config.REPORTS_DIR / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, indent=2)

    print(f"[train] modelo guardado en {config.MODEL_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
