# Pipeline reproducible de clasificación — Dataset Iris

![pipeline](https://github.com/SantiagoGR16/iris-pipeline/actions/workflows/pipeline.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11%2B-blue)

Pipeline de *machine learning* reproducible que **valida los datos, entrena, valida el modelo,
hace inferencia gobernada por un Contrato de Datos y ejecuta pruebas automatizadas**, todo con un
solo comando.

- **Autor:** Santiago Giraldo Rico
- **Curso:** Seminario de Investigación — Ciencia de Datos
- **Fecha:** noviembre de 2026
- **Licencia:** MIT

---

## Tabla de contenido
1. [Descripción](#1-descripción)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Requisitos previos](#3-requisitos-previos)
4. [Reproducir el pipeline desde cero](#4-reproducir-el-pipeline-desde-cero)
5. [Comandos disponibles](#5-comandos-disponibles)
6. [Etapas del pipeline](#6-etapas-del-pipeline)
7. [Validación de datos](#7-validación-de-datos)
8. [Contrato de Datos](#8-contrato-de-datos)
9. [Pruebas automatizadas](#9-pruebas-automatizadas)
10. [Resultados y evidencia de ejecución](#10-resultados-y-evidencia-de-ejecución)
11. [Solución de problemas](#11-solución-de-problemas)

---

## 1. Descripción

| Elemento | Detalle |
|---|---|
| Dataset | Iris (Fisher, 1936): 150 flores, 4 medidas en cm, 3 especies balanceadas (50 c/u) |
| Fuente | `data/raw/iris.csv`, formato de *scikit-learn* (1.ª fila = metadatos) |
| Tarea | Clasificación multiclase: *setosa*, *versicolor*, *virginica* |
| Modelo | `StandardScaler` + `LogisticRegression` dentro de un `sklearn.Pipeline` |
| Partición | 80 % entrenamiento / 20 % prueba, estratificada, `random_state=42` |
| Validación del modelo | Validación cruzada estratificada (5 folds) + compuerta de calidad: accuracy en prueba ≥ 0.90 |

## 2. Estructura del repositorio

```
iris-pipeline/
├── .github/workflows/pipeline.yml  # CI: ejecuta "make all" en cada push
├── contracts/
│   └── data_contract.json          # Contrato de Datos oficial (req. 3.4)
├── data/raw/iris.csv               # Datos de entrada (inmutables)
├── models/                         # Modelo entrenado (generado, no versionado)
├── notebooks/
│   └── pipeline_iris.ipynb         # Recorrido comentado y ejecutado del pipeline
├── reports/                        # Métricas y log de evidencia (generados)
├── src/
│   ├── config.py                   # Rutas y parámetros centralizados
│   ├── data_loader.py              # Carga del CSV
│   ├── validate.py                 # Validación de datos SIN librerías externas (req. 3.3)
│   ├── train.py                    # Entrenamiento
│   ├── evaluate.py                 # Validación del modelo + compuerta de calidad
│   └── predict.py                  # Inferencia validada contra el contrato
├── tests/                          # Pruebas automatizadas (pytest)
├── Makefile                        # Automatización (req. 3.2)
├── run_pipeline.py                 # Orquestador multiplataforma (req. 3.2)
├── requirements.txt
└── README.md
```

## 3. Requisitos previos

- **Python 3.11 o superior** (probado con 3.13; requirements ajustados para 3.14) (`python --version`)
- **Git** (`git --version`)
- **make** *(opcional)*: viene en Linux y macOS. En Windows se puede usar
  `python run_pipeline.py`, que hace exactamente lo mismo.

## 4. Reproducir el pipeline desde cero

```bash
# 1. Clonar el repositorio
git clone https://github.com/SantiagoGR16/iris-pipeline.git
cd iris-pipeline

# 2. Crear y activar un entorno virtual
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows (PowerShell o CMD)

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar TODO el pipeline con un solo comando
make all                           # Linux / macOS / Git Bash
# python run_pipeline.py           # Windows o cualquier sistema sin make
```

Resultado esperado: el log termina con `PIPELINE COMPLETADO` y los 21 tests en `PASSED`.
Se generan:

| Archivo | Contenido |
|---|---|
| `models/iris_model.joblib` | Modelo entrenado + conjunto de prueba |
| `reports/validation_report.json` | Resultado de la validación de datos |
| `reports/metrics.json` | Métricas de CV y prueba, matriz de confusión, compuerta |
| `reports/pipeline_run.log` | Evidencia completa de la ejecución |

Para ver el notebook: `pip install jupyterlab` y luego `jupyter lab notebooks/pipeline_iris.ipynb`.

## 5. Comandos disponibles

| Comando | Qué hace |
|---|---|
| `make all` / `python run_pipeline.py` | Pipeline completo: validar → entrenar → evaluar → predecir → probar |
| `python run_pipeline.py --skip-tests` | Pipeline sin pruebas |
| `make validate` / `python -m src.validate [archivo.csv]` | Solo valida datos |
| `make train` | Valida y entrena |
| `make evaluate` | Valida, entrena y evalúa el modelo |
| `make predict` / `python -m src.predict '<json>'` | Inferencia con el contrato |
| `make test` / `python -m pytest` | Pruebas automatizadas |
| `make clean` | Borra modelos y reportes generados |

Ejemplo de inferencia con un registro propio:

```bash
python -m src.predict '{"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.6, "petal_width": 2.2}'
```

## 6. Etapas del pipeline

```
data/raw/iris.csv
      │
      ▼
[1] validate.py ──✗──► se detiene (código 1) + reports/validation_report.json
      │ OK
      ▼
[2] train.py ──► split estratificado 80/20 ──► CV 5 folds ──► models/iris_model.joblib
      │
      ▼
[3] evaluate.py ──► accuracy < 0.90 ──✗──► se detiene
      │ OK                              
      ▼
[4] predict.py ──► valida cada registro con el contrato ──► especie o lista de errores
      │
      ▼
[5] pytest ──► 21 pruebas
```

Cada paso es un proceso independiente; si alguno termina con código distinto de 0, el
orquestador detiene la ejecución y lo deja registrado en el log.

## 7. Validación de datos

`src/validate.py` usa **únicamente la biblioteca estándar** (`csv`, `json`, `math`), sin pandas
ni librerías de validación. Las reglas se leen del contrato.

| Verificación | Cómo se valida |
|---|---|
| **Tipos de datos** | Variables convertibles a `float` y finitas; etiqueta entera. En inferencia se exige `int`/`float` real (se rechazan textos y booleanos). |
| **Valores nulos** | Se rechazan `""`, `NA`, `N/A`, `NaN`, `null`, `None`, `?` y `None` de Python. |
| **Rangos lógicos** | Cada variable dentro de `[min, max]` del contrato; etiqueta en `{0, 1, 2}`. |
| **Estructura** | 5 columnas por fila; encabezado coherente con el número de filas y las clases. |
| **Balance** (aviso) | Advierte si alguna clase no tiene 50 registros. |

## 8. Contrato de Datos

`contracts/data_contract.json` es la especificación oficial para la inferencia: campos requeridos,
tipo, unidad, rango permitido, prohibición de campos adicionales y de nulos, y el formato de salida.

**Entrada válida** → predicción `setosa`:

```json
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
```

**Entrada inválida** → rechazada con 5 errores:

```json
{"sepal_length": "cinco", "sepal_width": null, "petal_length": 12.0, "petal_width": -0.3, "color": "morado"}
```

Los rangos se fijaron con un margen alrededor de los mínimos y máximos observados en el dataset.
Para cambiar una regla basta con editar el JSON (y subir su `version`).

## 9. Pruebas automatizadas

21 pruebas en `tests/`:

- `test_validation.py` (15): dataset original válido; detección de tipos incorrectos, nulos,
  valores fuera de rango, clases no permitidas, columnas faltantes, encabezado inconsistente y
  archivo inexistente; ejemplos válido e inválido del contrato.
- `test_pipeline.py` (6): generación de artefactos, compuerta de calidad, partición
  estratificada, **reproducibilidad** (dos entrenamientos → mismas métricas) e inferencia.

Además, GitHub Actions (`.github/workflows/pipeline.yml`) ejecuta `make all` en un servidor limpio
en cada *push* y publica los reportes como artefacto descargable.

## 10. Resultados y evidencia de ejecución

| Métrica | Valor |
|---|---|
| Accuracy validación cruzada (5 folds) | 0.9583 ± 0.0264 |
| Accuracy en prueba (30 flores) | **0.9333** |
| F1 macro en prueba | 0.9333 |
| Compuerta de calidad (≥ 0.90) | Superada |

Matriz de confusión en prueba (filas = real):

| | setosa | versicolor | virginica |
|---|---|---|---|
| **setosa** | 10 | 0 | 0 |
| **versicolor** | 0 | 9 | 1 |
| **virginica** | 0 | 1 | 9 |

La evidencia completa está en [`reports/pipeline_run.log`](reports/pipeline_run.log) y en las
salidas del notebook.

## 11. Solución de problemas

| Problema | Solución |
|---|---|
| `make: command not found` (Windows) | Use `python run_pipeline.py` |
| `ModuleNotFoundError: No module named 'src'` | Ejecute los comandos desde la raíz del repositorio con `python -m src.<modulo>` |
| `python` no se reconoce | Pruebe con `python3` o `py` (Windows), o `make all PYTHON=py` |
| Fallan las dependencias | Actualice pip: `python -m pip install --upgrade pip` |

---

*Referencia del dataset:* Fisher, R. A. (1936). The use of multiple measurements in taxonomic
problems. *Annals of Eugenics, 7*(2), 179–188.
