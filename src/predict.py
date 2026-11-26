"""
predict.py
----------
Inferencia gobernada por el Contrato de Datos.

Antes de predecir, cada registro se valida contra contracts/data_contract.json.
Si no cumple el contrato, NO se envía al modelo y se devuelven los errores.

Uso:
    python -m src.predict                         # usa los ejemplos del contrato
    python -m src.predict '{"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}'
"""
import json
import sys

import joblib
import pandas as pd

from src import config
from src.validate import load_contract, validate_record


def predict_record(record, contract=None, bundle=None):
    """Valida y predice un registro. Devuelve un dict con el resultado."""
    contract = contract or load_contract()
    errors = validate_record(record, contract)
    if errors:
        return {"status": "rejected", "errors": errors}

    bundle = bundle or joblib.load(config.MODEL_PATH)
    labels = contract["training_schema"]["target_labels"]
    X = pd.DataFrame([record], columns=bundle["features"])
    pred = int(bundle["model"].predict(X)[0])
    proba = bundle["model"].predict_proba(X)[0]
    return {
        "status": "ok",
        "species": labels[str(pred)],
        "probabilities": {labels[str(i)]: round(float(p), 4) for i, p in enumerate(proba)},
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    contract = load_contract()

    if argv:
        records = {"entrada_usuario": json.loads(argv[0])}
    else:
        ex = contract["examples"]
        records = {"ejemplo_valido": ex["valid_input"],
                   "ejemplo_invalido": ex["invalid_input"]}

    for name, record in records.items():
        result = predict_record(record, contract)
        print(f"[predict] {name}: {json.dumps(record, ensure_ascii=False)}")
        print(f"[predict]   -> {json.dumps(result, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
