"""
run_pipeline.py
---------------
Script de orquestación (REQUISITO 3.2): ejecuta todo el pipeline con UN comando.

    python run_pipeline.py              # validar -> entrenar -> evaluar -> pruebas
    python run_pipeline.py --skip-tests # sin pruebas automatizadas

Funciona igual en Windows, macOS y Linux (no requiere 'make').
Cada paso se ejecuta como un proceso independiente; si uno falla
(código de salida distinto de 0), el pipeline se detiene ahí.

Todo lo que se imprime queda también guardado en reports/pipeline_run.log
como evidencia de ejecución.
"""
import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "reports" / "pipeline_run.log"

# Pasos del pipeline en orden: (nombre, comando)
STEPS = [
    ("1. Validación de datos", [sys.executable, "-m", "src.validate"]),
    ("2. Entrenamiento", [sys.executable, "-m", "src.train"]),
    ("3. Validación del modelo", [sys.executable, "-m", "src.evaluate"]),
    ("4. Inferencia con contrato", [sys.executable, "-m", "src.predict"]),
    ("5. Pruebas automatizadas", [sys.executable, "-m", "pytest", "tests", "-v"]),
]


def log(msg, fh):
    """Imprime en pantalla y escribe en el archivo de log."""
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def main():
    parser = argparse.ArgumentParser(description="Pipeline reproducible Iris")
    parser.add_argument("--skip-tests", action="store_true", help="omite pytest")
    args = parser.parse_args()

    steps = STEPS[:-1] if args.skip_tests else STEPS
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        log("=" * 64, fh)
        log(f"PIPELINE IRIS | inicio: {datetime.now():%Y-%m-%d %H:%M:%S}", fh)
        log(f"Python {platform.python_version()} | {platform.system()} {platform.release()}", fh)
        log("=" * 64, fh)

        for name, cmd in steps:
            log(f"\n>>> {name}", fh)
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            for line in (result.stdout + result.stderr).rstrip().splitlines():
                log(f"    {line}", fh)
            if result.returncode != 0:
                log(f"\n[X] El paso '{name}' falló (código {result.returncode}). "
                    "Pipeline detenido.", fh)
                return result.returncode
            log(f"[OK] {name}", fh)

        log(f"\nPIPELINE COMPLETADO en {time.time() - start:.1f} s", fh)
        log(f"Evidencia guardada en {LOG_PATH.relative_to(ROOT)}", fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
