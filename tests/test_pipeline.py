"""
Pruebas de extremo a extremo: entrenamiento, validación del modelo,
reproducibilidad e inferencia gobernada por el contrato.
"""
import json

import joblib
import pytest

from src import config, evaluate, train
from src.predict import predict_record


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """
    Entrena y evalúa una vez con los módulos src/train y src/evaluate.
    Las rutas de salida se redirigen a una carpeta temporal para NO
    sobrescribir el modelo y las métricas que generó run_pipeline.py.
    """
    tmp = tmp_path_factory.mktemp("artefactos")
    originales = (config.MODELS_DIR, config.REPORTS_DIR, config.MODEL_PATH, config.METRICS_PATH)
    config.MODELS_DIR, config.REPORTS_DIR = tmp, tmp
    config.MODEL_PATH, config.METRICS_PATH = tmp / "model.joblib", tmp / "metrics.json"
    try:
        assert train.main() == 0
        assert evaluate.main() == 0
        yield joblib.load(config.MODEL_PATH)
    finally:  # se restauran las rutas reales aunque algo falle
        config.MODELS_DIR, config.REPORTS_DIR, config.MODEL_PATH, config.METRICS_PATH = originales


def test_se_generan_artefactos(trained):
    assert config.MODEL_PATH.exists()
    assert config.METRICS_PATH.exists()


def test_modelo_supera_umbral(trained):
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics["test_accuracy"] >= config.MIN_TEST_ACCURACY
    assert metrics["quality_gate"]["passed"] is True


def test_particion_estratificada(trained):
    # 30 registros de prueba, 10 por especie
    assert len(trained["y_test"]) == 30
    assert trained["y_test"].value_counts().tolist() == [10, 10, 10]


def test_reproducibilidad(trained):
    """Entrenar dos veces con la misma semilla produce las mismas métricas."""
    first = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    train.main()
    evaluate.main()
    second = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    assert first["test_accuracy"] == second["test_accuracy"]
    assert first["cv_scores"] == second["cv_scores"]


def test_prediccion_ejemplo_valido(trained, contract):
    result = predict_record(contract["examples"]["valid_input"], contract, trained)
    assert result["status"] == "ok"
    assert result["species"] == contract["examples"]["valid_input_expected_output"]["species"]


def test_prediccion_rechaza_ejemplo_invalido(trained, contract):
    result = predict_record(contract["examples"]["invalid_input"], contract, trained)
    assert result["status"] == "rejected"
    assert len(result["errors"]) == 5
