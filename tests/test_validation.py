"""
Pruebas de la lógica de validación (src/validate.py) y del Contrato de Datos.
Cubren los tres criterios del requisito 3.3: tipos, nulos y rangos.
"""
from src import config
from src.validate import validate_record, validate_training_file


# ---------------------------------------------------------------------------
# Archivo real
# ---------------------------------------------------------------------------
def test_dataset_original_es_valido():
    report = validate_training_file(config.RAW_DATA_PATH)
    assert report["status"] == "OK", report["errors"]
    assert report["valid_rows"] == 150
    assert report["class_counts"] == {"0": 50, "1": 50, "2": 50}


# ---------------------------------------------------------------------------
# Validación de archivo: tipos, nulos, rangos, estructura
# ---------------------------------------------------------------------------
def test_detecta_tipo_incorrecto(make_csv):
    report = validate_training_file(make_csv("5.1,abc,1.4,0.2,0"))
    assert report["status"] == "FAILED"
    assert any("sepal_width" in e and "tipo inválido" in e for e in report["errors"])


def test_detecta_etiqueta_no_entera(make_csv):
    report = validate_training_file(make_csv("5.1,3.5,1.4,0.2,1.5"))
    assert any("target" in e and "tipo inválido" in e for e in report["errors"])


def test_detecta_nulos(make_csv):
    report = validate_training_file(make_csv("5.1,,1.4,0.2,0\n4.9,3.0,NaN,0.2,0"))
    assert report["n_errors"] == 2
    assert all("valor nulo" in e for e in report["errors"])


def test_detecta_fuera_de_rango(make_csv):
    report = validate_training_file(make_csv("15.0,3.5,1.4,-0.2,0"))
    assert any("sepal_length" in e and "fuera de rango" in e for e in report["errors"])
    assert any("petal_width" in e and "fuera de rango" in e for e in report["errors"])


def test_detecta_clase_no_permitida(make_csv):
    report = validate_training_file(make_csv("5.1,3.5,1.4,0.2,7"))
    assert any("no está en" in e for e in report["errors"])


def test_detecta_columnas_faltantes(make_csv):
    report = validate_training_file(make_csv("5.1,3.5,1.4,0"))
    assert any("columnas" in e for e in report["errors"])


def test_detecta_encabezado_inconsistente(make_csv):
    path = make_csv("5.1,3.5,1.4,0.2,0", header="99,4,setosa,versicolor,virginica\n")
    report = validate_training_file(path)
    assert any("encabezado declara 99 filas" in e for e in report["errors"])


def test_archivo_inexistente():
    report = validate_training_file("no_existe.csv")
    assert report["status"] == "FAILED"


# ---------------------------------------------------------------------------
# Contrato de datos (requisito 3.4)
# ---------------------------------------------------------------------------
def test_contrato_tiene_ejemplos(contract):
    assert "valid_input" in contract["examples"]
    assert "invalid_input" in contract["examples"]


def test_ejemplo_valido_del_contrato_pasa(contract):
    assert validate_record(contract["examples"]["valid_input"], contract) == []


def test_ejemplo_invalido_del_contrato_falla(contract):
    errors = validate_record(contract["examples"]["invalid_input"], contract)
    # Los errores obtenidos deben coincidir con los documentados en el contrato
    assert sorted(errors) == sorted(contract["examples"]["invalid_input_expected_errors"])


def test_registro_rechaza_booleanos_y_texto(contract):
    record = {"sepal_length": True, "sepal_width": "3.5",
              "petal_length": 1.4, "petal_width": 0.2}
    errors = validate_record(record, contract)
    assert len(errors) == 2


def test_registro_rechaza_campo_faltante(contract):
    errors = validate_record({"sepal_length": 5.1, "sepal_width": 3.5,
                              "petal_length": 1.4}, contract)
    assert errors == ["campo requerido ausente: petal_width"]


def test_registro_acepta_enteros(contract):
    record = {"sepal_length": 6, "sepal_width": 3, "petal_length": 5, "petal_width": 2}
    assert validate_record(record, contract) == []
