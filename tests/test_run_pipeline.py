"""
Pruebas de las etapas de run_pipeline.py:
partición 80/20, control de calidad y modelo almacenado por la etapa 3.
"""
import json

import joblib
import pytest

import run_pipeline as rp
from src import config


@pytest.fixture(scope="module")
def splits():
    """Ejecuta la etapa 1 (no necesita MLflow) y devuelve train y test."""
    return rp.stage_1_prepare()


def test_particion_80_20(splits):
    train_df, test_df = splits
    assert len(train_df) == 120 and len(test_df) == 30
    assert len(test_df) / (len(train_df) + len(test_df)) == pytest.approx(0.20)


def test_particion_estratificada_y_sin_fugas(splits):
    train_df, test_df = splits
    assert test_df[config.TARGET].value_counts().tolist() == [10, 10, 10]
    assert not set(train_df["row_id"]) & set(test_df["row_id"])


def test_archivos_procesados_existen(splits):
    assert rp.TRAIN_PATH.exists() and rp.TEST_PATH.exists()


def test_etapa_2_aprueba_datos_correctos(splits):
    rp.stage_2_validate(*splits)          # no debe lanzar excepción


def test_etapa_2_detecta_nulos_y_rangos(splits):
    train_df, test_df = (s.copy() for s in splits)
    train_df.iloc[0, train_df.columns.get_loc("sepal_width")] = None
    test_df.iloc[0, test_df.columns.get_loc("petal_length")] = 40.0
    with pytest.raises(rp.StageError):
        rp.stage_2_validate(train_df, test_df)


def test_etapa_2_detecta_fuga(splits):
    train_df, test_df = splits
    with pytest.raises(rp.StageError):
        rp.stage_2_validate(train_df, test_df.assign(row_id=train_df["row_id"].iloc[:30].values))


def test_etapa_3_modelo_almacenado_y_metricas():
    """Verifica lo que dejó la etapa 3 (se omite si aún no se ha corrido el pipeline)."""
    if not config.METRICS_PATH.exists() or not config.MODEL_PATH.exists():
        pytest.skip("ejecute primero: python run_pipeline.py")
    metrics = json.loads(config.METRICS_PATH.read_text())
    assert {"accuracy", "f1_score"} <= metrics.keys()
    assert metrics["accuracy"] >= config.MIN_TEST_ACCURACY
    bundle = joblib.load(config.MODEL_PATH)
    assert hasattr(bundle["model"], "predict")
