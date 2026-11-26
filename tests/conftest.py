"""Configuración compartida de pytest: agrega la raíz del proyecto al path."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.validate import load_contract  # noqa: E402

HEADER = "150,4,setosa,versicolor,virginica\n"


@pytest.fixture(scope="session")
def contract():
    """Contrato de datos cargado una sola vez para todas las pruebas."""
    return load_contract()


@pytest.fixture
def make_csv(tmp_path):
    """
    Fábrica de CSV temporales: recibe el cuerpo (sin encabezado) y devuelve
    la ruta. Ajusta automáticamente el número de filas del encabezado.
    """
    def _make(body, header=None):
        lines = [l for l in body.strip().splitlines() if l.strip()]
        head = header or f"{len(lines)},4,setosa,versicolor,virginica\n"
        path = tmp_path / "datos.csv"
        path.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
        return path
    return _make
