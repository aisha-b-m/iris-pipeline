"""
evaluate.py
-----------
Validación del modelo: lo evalúa sobre el conjunto de prueba (datos que
nunca vio durante el entrenamiento) y aplica una compuerta de calidad.

Si la exactitud (accuracy) en prueba es menor que MIN_TEST_ACCURACY,
el script termina con código 1 y el pipeline se marca como fallido.

Uso:
    python -m src.evaluate
"""
import json
import sys

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src import config

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def main():
    if not config.MODEL_PATH.exists():
        print("[evaluate] no existe el modelo. Ejecute primero: python -m src.train")
        return 1

    bundle = joblib.load(config.MODEL_PATH)
    model, X_test, y_test = bundle["model"], bundle["X_test"], bundle["y_test"]

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print(f"[evaluate] accuracy en prueba: {acc:.4f}")
    print(f"[evaluate] F1 macro en prueba: {f1:.4f}")
    print("[evaluate] matriz de confusión (filas = real, columnas = predicho):")
    for name, row in zip(CLASS_NAMES, cm):
        print(f"           {name:<11} {row.tolist()}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    # Se combinan las métricas de entrenamiento y prueba en un solo archivo
    train_path = config.REPORTS_DIR / "train_metrics.json"
    train_metrics = json.loads(train_path.read_text(encoding="utf-8")) if train_path.exists() else {}
    passed = acc >= config.MIN_TEST_ACCURACY
    metrics = {
        **train_metrics,
        "accuracy": round(float(acc), 4),        # mismas claves que escribe run_pipeline.py
        "f1_score": round(float(f1), 4),
        "test_accuracy": round(float(acc), 4),
        "test_f1_macro": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "quality_gate": {"min_test_accuracy": config.MIN_TEST_ACCURACY,
                         "passed": bool(passed)},
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if not passed:
        print(f"[evaluate] FALLÓ la compuerta: {acc:.4f} < {config.MIN_TEST_ACCURACY}")
        return 1
    print(f"[evaluate] compuerta de calidad superada (>= {config.MIN_TEST_ACCURACY})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
