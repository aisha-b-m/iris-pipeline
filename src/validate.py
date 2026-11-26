"""
validate.py
-----------
Lógica de validación de datos de entrada.

REQUISITO 3.3: esta validación se implementa SIN librerías externas.
Solo se usa la biblioteca estándar de Python (csv, json, math, sys, pathlib).
No se usa pandas, numpy, pydantic, great_expectations, etc.

Verifica tres cosas, tanto para el archivo de entrenamiento como para
registros individuales de inferencia:
    1. Tipos de datos correctos.
    2. Ausencia de valores nulos.
    3. Rangos lógicos de cada variable.

Las reglas NO están escritas aquí: se leen del Contrato de Datos
(contracts/data_contract.json), que es la única fuente de verdad.

Uso por consola:
    python -m src.validate                  # valida data/raw/iris.csv
    python -m src.validate ruta/otro.csv    # valida otro archivo
"""
import csv
import json
import math
import sys
from pathlib import Path

# Permite ejecutar el archivo tanto con "python -m src.validate" como directo
try:
    from src import config
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config

# Textos que se interpretan como "valor nulo" cuando vienen dentro de un CSV
NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def load_contract(path=config.CONTRACT_PATH):
    """Lee el contrato JSON y lo devuelve como diccionario."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_null(value):
    """True si el valor representa un nulo (None, NaN o texto vacío/'NA'...)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
        return True
    return False


def _parse_csv_number(text, expected_type):
    """
    Convierte un texto del CSV al tipo esperado ('float' o 'int').
    Devuelve (valor, mensaje_error). Si todo sale bien, el error es None.
    """
    text = text.strip()
    try:
        if expected_type == "int":
            # "1" es válido, "1.0" o "1.5" no lo son para la etiqueta
            if not text.lstrip("-").isdigit():
                return None, f"tipo inválido (se esperaba int, llegó '{text}')"
            return int(text), None
        value = float(text)
    except ValueError:
        return None, f"tipo inválido (se esperaba {expected_type}, llegó '{text}')"
    if not math.isfinite(value):
        return None, f"valor no finito ('{text}')"
    return value, None


def _check_range(name, value, rule):
    """Devuelve un mensaje de error si el valor está fuera de [min, max]."""
    if value < rule["min"] or value > rule["max"]:
        return f"{name}: {value} fuera de rango [{rule['min']}, {rule['max']}]"
    return None


# ---------------------------------------------------------------------------
# 1) Validación de UN registro (inferencia)
# ---------------------------------------------------------------------------
def validate_record(record, contract=None):
    """
    Valida un registro (diccionario) que se enviará al modelo para predecir.
    En inferencia somos estrictos: los números deben llegar como int/float,
    no como texto. Devuelve una lista de errores (vacía = registro válido).
    """
    contract = contract or load_contract()
    schema = contract["input_schema"]
    errors = []

    if not isinstance(record, dict):
        return [f"el registro debe ser un objeto JSON/dict, llegó {type(record).__name__}"]

    # Campos faltantes
    for field in schema["required_fields"]:
        if field not in record:
            errors.append(f"campo requerido ausente: {field}")

    # Campos sobrantes (el contrato no los permite)
    if not schema.get("additional_fields_allowed", False):
        for field in record:
            if field not in schema["fields"]:
                errors.append(f"campo no permitido por el contrato: {field}")

    # Revisión campo por campo: nulos -> tipo -> rango
    for field, rule in schema["fields"].items():
        if field not in record:
            continue
        value = record[field]

        if _is_null(value):
            errors.append(f"{field}: valor nulo no permitido")
            continue

        # bool es subclase de int en Python; lo rechazamos explícitamente
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(
                f"{field}: tipo inválido (se esperaba {rule['type']}, "
                f"llegó {type(value).__name__})"
            )
            continue

        if not math.isfinite(value):
            errors.append(f"{field}: valor no finito")
            continue

        range_error = _check_range(field, float(value), rule)
        if range_error:
            errors.append(range_error)

    return errors


# ---------------------------------------------------------------------------
# 2) Validación del ARCHIVO de entrenamiento
# ---------------------------------------------------------------------------
def validate_training_file(path=config.RAW_DATA_PATH, contract=None):
    """
    Valida el CSV de entrenamiento completo y devuelve un reporte (dict).

    Formato esperado del archivo (formato de sklearn):
        fila 1 -> n_muestras, n_variables, nombre_clase_0, nombre_clase_1, ...
        fila 2+ -> 4 variables numéricas + etiqueta entera (0, 1 o 2)
    """
    contract = contract or load_contract()
    features = contract["input_schema"]["required_fields"]
    rules = contract["input_schema"]["fields"]
    tschema = contract["training_schema"]
    n_cols = len(features) + 1

    errors = []     # problemas que invalidan el archivo
    warnings = []   # avisos que no detienen el pipeline
    valid_rows = 0
    class_counts = {}

    path = Path(path)
    if not path.exists():
        return _report(path, 0, 0, [f"archivo no encontrado: {path}"], [], {})

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # Se descartan líneas totalmente vacías (p. ej. salto de línea final)
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if not rows:
        return _report(path, 0, 0, ["el archivo está vacío"], [], {})

    # --- Encabezado ---
    header, data = rows[0], rows[1:]
    try:
        declared_rows = int(header[0])
        declared_features = int(header[1])
        class_names = [c.strip() for c in header[2:]]
    except (ValueError, IndexError):
        errors.append(f"encabezado con formato inesperado: {header}")
        declared_rows, declared_features, class_names = None, None, []

    if declared_features is not None and declared_features != len(features):
        errors.append(
            f"encabezado declara {declared_features} variables, "
            f"el contrato exige {len(features)}"
        )
    expected_names = [tschema["target_labels"][str(v)] for v in tschema["target_allowed_values"]]
    if class_names and class_names != expected_names:
        errors.append(f"nombres de clase {class_names} no coinciden con {expected_names}")
    if declared_rows is not None and declared_rows != len(data):
        errors.append(f"encabezado declara {declared_rows} filas pero hay {len(data)}")
    if len(data) != tschema["expected_rows"]:
        warnings.append(f"se esperaban {tschema['expected_rows']} filas, hay {len(data)}")

    # --- Filas de datos ---
    for i, row in enumerate(data, start=2):   # start=2 -> número de línea real
        row_errors = []

        if len(row) != n_cols:
            errors.append(f"línea {i}: tiene {len(row)} columnas, se esperaban {n_cols}")
            continue

        # Variables predictoras
        for name, cell in zip(features, row[:-1]):
            if _is_null(cell):
                row_errors.append(f"{name}: valor nulo")
                continue
            value, err = _parse_csv_number(cell, rules[name]["type"])
            if err:
                row_errors.append(f"{name}: {err}")
                continue
            range_error = _check_range(name, value, rules[name])
            if range_error:
                row_errors.append(range_error)

        # Variable objetivo
        cell = row[-1]
        if _is_null(cell):
            row_errors.append(f"{config.TARGET}: valor nulo")
        else:
            value, err = _parse_csv_number(cell, tschema["target_type"])
            if err:
                row_errors.append(f"{config.TARGET}: {err}")
            elif value not in tschema["target_allowed_values"]:
                row_errors.append(
                    f"{config.TARGET}: {value} no está en {tschema['target_allowed_values']}"
                )
            else:
                class_counts[value] = class_counts.get(value, 0) + 1

        if row_errors:
            errors.extend(f"línea {i}: {e}" for e in row_errors)
        else:
            valid_rows += 1

    # --- Balance de clases (aviso, no error) ---
    expected_per_class = tschema.get("expected_rows_per_class")
    for label in tschema["target_allowed_values"]:
        n = class_counts.get(label, 0)
        if expected_per_class and n != expected_per_class:
            warnings.append(f"clase {label} tiene {n} filas (se esperaban {expected_per_class})")

    return _report(path, len(data), valid_rows, errors, warnings, class_counts)


def _rel(path):
    """Ruta relativa a la raíz del proyecto (si aplica), para reportes portables."""
    try:
        return str(Path(path).resolve().relative_to(config.ROOT_DIR))
    except ValueError:
        return str(path)


def _report(path, total, valid, errors, warnings, class_counts):
    """Arma el reporte de validación con una estructura fija."""
    return {
        "file": _rel(path),
        "status": "OK" if not errors else "FAILED",
        "total_rows": total,
        "valid_rows": valid,
        "n_errors": len(errors),
        "errors": errors,
        "warnings": warnings,
        "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
    }


# ---------------------------------------------------------------------------
# Punto de entrada por consola
# ---------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else config.RAW_DATA_PATH

    report = validate_training_file(path)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[validate] archivo: {report['file']}")
    print(f"[validate] filas válidas: {report['valid_rows']}/{report['total_rows']}")
    print(f"[validate] conteo por clase: {report['class_counts']}")
    for w in report["warnings"]:
        print(f"[validate] AVISO: {w}")
    for e in report["errors"][:20]:
        print(f"[validate] ERROR: {e}")
    print(f"[validate] estado: {report['status']}")

    # Código de salida != 0 detiene el pipeline (Makefile / orquestador)
    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
