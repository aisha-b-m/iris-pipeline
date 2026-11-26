# Pipeline reproducible de clasificación — Dataset Iris

![pipeline](https://github.com/SantiagoGR16/iris-pipeline/actions/workflows/pipeline.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![licencia](https://img.shields.io/badge/licencia-MIT-green)

Pipeline de *machine learning* reproducible que, con **un solo comando**, prepara y parte los
datos (80/20), los valida contra un **Contrato de Datos**, entrena un clasificador, registra el
experimento en **MLflow** y ejecuta **pruebas automatizadas**.

| | |
|---|---|
| **Autor** | Santiago Giraldo Rico |
| **Curso** | Seminario de Investigación — Ciencia de Datos |
| **Fecha** | noviembre de 2026 |
| **Licencia** | MIT (ver `LICENSE`) |

---

## Tabla de contenido

1. [Descripción](#1-descripción)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Requisitos previos](#3-requisitos-previos)
4. [Reproducir el pipeline desde cero](#4-reproducir-el-pipeline-desde-cero)
5. [Verificar que todo funcionó](#5-verificar-que-todo-funcionó)
6. [Etapas del pipeline](#6-etapas-del-pipeline)
7. [Validación de datos](#7-validación-de-datos)
8. [Contrato de Datos](#8-contrato-de-datos)
9. [Pruebas automatizadas](#9-pruebas-automatizadas)
10. [Comandos disponibles](#10-comandos-disponibles)
11. [Resultados](#11-resultados)
12. [Solución de problemas](#12-solución-de-problemas)
13. [Decisiones de reproducibilidad](#13-decisiones-de-reproducibilidad)

---

## 1. Descripción

| Elemento | Detalle |
|---|---|
| Dataset | Iris (Fisher, 1936): 150 flores, 4 medidas en cm, 3 especies balanceadas (50 c/u) |
| Archivo | `data/raw/iris.csv`, formato de *scikit-learn*: la 1.ª fila son metadatos (`150,4,setosa,versicolor,virginica`) y la última columna es la especie codificada (0, 1, 2) |
| Tarea | Clasificación multiclase: *setosa*, *versicolor*, *virginica* |
| Modelo | `StandardScaler` + `LogisticRegression` dentro de un `sklearn.Pipeline` |
| Partición | 80 % entrenamiento (120) / 20 % prueba (30), estratificada, `random_state=42` |
| Métricas | *Accuracy* y *F1 macro* sobre el conjunto de prueba |
| Compuerta de calidad | El pipeline falla si la *accuracy* en prueba es menor que 0.90 |

## 2. Estructura del repositorio

```
iris-pipeline/
├── .github/workflows/pipeline.yml  # Integración continua: corre el pipeline en cada push
├── contracts/
│   └── data_contract.json          # Contrato de Datos oficial para inferencia
├── data/
│   ├── raw/iris.csv                # Datos originales (no se modifican)
│   └── processed/                  # train.csv y test.csv (generados por la etapa 1)
├── models/                         # Modelo entrenado (generado, no se versiona)
├── notebooks/
│   └── pipeline_iris.ipynb         # Recorrido comentado del pipeline, paso a paso
├── reports/                        # Métricas y log de evidencia (generados)
├── src/
│   ├── config.py                   # Rutas y parámetros centralizados
│   ├── data_loader.py              # Carga del CSV
│   ├── validate.py                 # Validación de datos SIN librerías externas
│   ├── train.py                    # Entrenamiento (con validación cruzada)
│   ├── evaluate.py                 # Evaluación del modelo + compuerta de calidad
│   └── predict.py                  # Inferencia validada contra el contrato
├── tests/                          # Pruebas automatizadas (pytest)
├── run_pipeline.py                 # Script de automatización: las 4 etapas con un comando
├── Makefile                        # Atajos para Linux / macOS
├── requirements.txt                # Dependencias
├── pytest.ini                      # Configuración de pytest
├── LICENSE
└── README.md
```

> **Importante:** el proyecto funciona como un todo. `run_pipeline.py` importa los módulos de
> `src/` y lee `contracts/`, `data/` y `tests/`. **No copie archivos sueltos**: clone o
> descomprima la carpeta completa y ejecute los comandos desde su raíz.

## 3. Requisitos previos

| Herramienta | Versión | Cómo verificar | Nota |
|---|---|---|---|
| Python | 3.11 o superior | `python --version` (Windows: `py --list`) | Probado con 3.13; dependencias ajustadas para 3.14 |
| Git | cualquiera reciente | `git --version` | Solo para clonar el repositorio |
| make | opcional | `make --version` | Viene en Linux/macOS. En Windows no hace falta |

**Instalar Python si no lo tiene (Windows):**

```powershell
winget install Python.Python.3.12
```

o descárguelo de [python.org](https://www.python.org/downloads/) y marque la casilla
*"Add python.exe to PATH"*. Cierre y vuelva a abrir la terminal después de instalar.

## 4. Reproducir el pipeline desde cero

Todos los comandos se ejecutan en una terminal (PowerShell en Windows; Terminal en macOS/Linux).

### Paso 1. Obtener el proyecto completo

```bash
git clone https://github.com/SantiagoGR16/iris-pipeline.git
cd iris-pipeline
```

Sin Git: en GitHub, botón **Code → Download ZIP**, extraiga el ZIP completo y entre a la carpeta
con `cd iris-pipeline`.

Compruebe que está en la carpeta correcta: `dir` (Windows) o `ls` (macOS/Linux) debe mostrar
`src`, `contracts`, `data`, `tests` y `run_pipeline.py`.

### Paso 2. Crear el entorno virtual

El entorno aísla las dependencias del proyecto del resto del sistema.

**Windows (PowerShell):**

```powershell
py --list                 # muestra las versiones de Python instaladas
py -3.14 -m venv .venv    # cambie 3.14 por una versión que aparezca en la lista
```

> Si sale `No suitable Python runtime found`, esa versión no está instalada.
> Use otra de las que muestra `py --list`, o simplemente `py -m venv .venv`.

**macOS / Linux:**

```bash
python3 -m venv .venv
```

### Paso 3. Activar el entorno

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

> Si aparece *"la ejecución de scripts está deshabilitada en este sistema"*, ejecute primero
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y vuelva a activar.
> En CMD (no PowerShell) use `.venv\Scripts\activate.bat`.

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Sabrá que está activo porque la terminal muestra `(.venv)` al inicio de la línea.

### Paso 4. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se instalan numpy, pandas, scikit-learn, joblib, pytest, matplotlib y MLflow.

### Paso 5. Ejecutar el pipeline completo (un solo comando)

```bash
python run_pipeline.py
```

En Linux/macOS también funciona `make all`, que hace lo mismo.

### Paso 6. Ver el experimento en MLflow (opcional)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abra <http://127.0.0.1:5000> en el navegador y entre al experimento **iris-pipeline**: ahí están
los parámetros, las métricas `accuracy` y `f1_score`, el modelo y el contrato de cada ejecución.
Para detener el servidor, presione `Ctrl + C` en la terminal.

### Paso 7. Abrir el notebook (opcional)

```bash
pip install jupyterlab
jupyter lab notebooks/pipeline_iris.ipynb
```

## 5. Verificar que todo funcionó

La salida debe terminar así:

```
[OK] Etapa 4

PIPELINE COMPLETADO en X.X s
```

y en la etapa 4 deben aparecer **28 pruebas `PASSED`**. Se generan estos archivos:

| Archivo | Contenido |
|---|---|
| `data/processed/train.csv` | 120 registros de entrenamiento (80 %) |
| `data/processed/test.csv` | 30 registros de prueba (20 %) |
| `models/iris_model.joblib` | Modelo entrenado |
| `reports/metrics.json` | *Accuracy* y *F1* en prueba |
| `mlflow.db`, `mlruns/` | Registro del experimento en MLflow |
| `reports/pipeline_run.log` | **Evidencia de ejecución**: copia completa de lo que se vio en pantalla |

Si una etapa falla, el pipeline se detiene, muestra `[X] PIPELINE DETENIDO` con el motivo y
termina con código de salida 1. Consulte la sección [12](#12-solución-de-problemas).

## 6. Etapas del pipeline

`run_pipeline.py` ejecuta 4 etapas en orden:

```
data/raw/iris.csv
      │
      ▼
[1] PREPARACIÓN DE DATOS
      │  • asigna nombres de columna y tipos
      │  • partición estratificada 80/20 (random_state=42)
      │  • guarda data/processed/train.csv y test.csv
      ▼
[2] VALIDACIÓN Y CONTROL DE CALIDAD ──✗──► se detiene
      │  • archivo crudo: tipos, nulos, rangos y estructura (src/validate.py)
      │  • cada registro de train y test contra el Contrato de Datos
      │  • proporción 80/20, sin registros compartidos (fuga), clases balanceadas
      ▼
[3] ENTRENAMIENTO Y REGISTRO EN MLFLOW ──✗──► se detiene si accuracy < 0.90
      │  • StandardScaler + LogisticRegression con el 80 %
      │  • accuracy y F1 macro sobre el 20 %
      │  • modelo en models/; parámetros, métricas, modelo y contrato en MLflow
      ▼
[4] PRUEBAS AUTOMATIZADAS ──✗──► se detiene si alguna falla
         • pytest: 28 pruebas
```

`python run_pipeline.py --skip-tests` ejecuta solo las etapas 1 a 3.

## 7. Validación de datos

`src/validate.py` usa **únicamente la biblioteca estándar de Python** (`csv`, `json`, `math`,
`sys`, `pathlib`): no usa pandas, numpy ni librerías de validación. Las reglas (tipos, rangos y
clases) se leen del Contrato de Datos, por lo que no están escritas en el código.

| Verificación | Cómo se valida | Función |
|---|---|---|
| **Tipos de datos** | Las 4 medidas deben ser números decimales finitos y la especie un entero. En inferencia se exige un `int`/`float` real: se rechazan textos (`"5.1"`) y booleanos | `_parse_csv_number()`, `validate_record()` |
| **Valores nulos** | Se rechazan `""`, `NA`, `N/A`, `NaN`, `null`, `None` y `?` | `_is_null()` |
| **Rangos lógicos** | Cada medida dentro del `[min, max]` del contrato; especie en `{0, 1, 2}` | `_check_range()` |
| **Estructura** | 5 columnas por fila; encabezado coherente con las filas y las clases | `validate_training_file()` |

Validar solo los datos:

```bash
python -m src.validate                  # valida data/raw/iris.csv
python -m src.validate otro_archivo.csv # valida otro archivo con el mismo formato
```

El resultado se guarda en `reports/validation_report.json`, con cada error y su número de línea.

## 8. Contrato de Datos

`contracts/data_contract.json` es la especificación oficial para la **inferencia futura** del
modelo. Define:

- los 4 campos obligatorios, su tipo (`float`), unidad (cm) y rango permitido;
- la prohibición de nulos y de campos adicionales;
- la salida del modelo (`species`: `setosa`, `versicolor` o `virginica`);
- un **ejemplo de entrada válida** y un **ejemplo de entrada inválida**, con los errores esperados.

| Campo | Mínimo | Máximo |
|---|---|---|
| `sepal_length` | 4.0 | 8.5 |
| `sepal_width` | 1.5 | 5.0 |
| `petal_length` | 0.5 | 7.5 |
| `petal_width` | 0.05 | 3.0 |

Los rangos tienen un margen alrededor de los valores mínimos y máximos observados en el dataset.

**Entrada válida** → predicción `setosa`:

```json
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
```

**Entrada inválida** → rechazada con 5 errores (tipo, nulo, dos rangos y un campo no permitido):

```json
{"sepal_length": "cinco", "sepal_width": null, "petal_length": 12.0, "petal_width": -0.3, "color": "morado"}
```

Probar la inferencia:

```bash
python -m src.predict    # usa los dos ejemplos del contrato (requiere haber corrido el pipeline)
```

Para cambiar una regla, edite el JSON y suba su campo `version`.

## 9. Pruebas automatizadas

28 pruebas en `tests/`, que se ejecutan en la etapa 4 o por separado con `python -m pytest -v`:

| Archivo | Pruebas | Qué verifica |
|---|---|---|
| `test_validation.py` | 15 | Dataset original válido; detección de tipos incorrectos, nulos, valores fuera de rango, clases no permitidas, columnas faltantes, encabezado inconsistente y archivo inexistente; ejemplos válido e inválido del contrato |
| `test_pipeline.py` | 6 | Artefactos generados, compuerta de calidad, partición estratificada, **reproducibilidad** (dos entrenamientos → mismas métricas) e inferencia |
| `test_run_pipeline.py` | 7 | Partición 80/20, estratificación, ausencia de fugas, detección de nulos/rangos/fugas en la etapa 2, y modelo y métricas de la etapa 3 |

Además, GitHub Actions (`.github/workflows/pipeline.yml`) ejecuta el pipeline completo en un
servidor limpio en cada *push* y publica los reportes como artefacto descargable (pestaña
**Actions** del repositorio).

## 10. Comandos disponibles

Ejecútelos desde la raíz del proyecto, con el entorno virtual activo.

| Comando | Qué hace |
|---|---|
| `python run_pipeline.py` | Pipeline completo de 4 etapas |
| `python run_pipeline.py --skip-tests` | Solo etapas 1 a 3 |
| `python -m src.validate` | Solo valida los datos |
| `python -m src.train` | Valida y entrena (incluye validación cruzada de 5 folds) |
| `python -m src.evaluate` | Evalúa el modelo entrenado y aplica la compuerta de calidad |
| `python -m src.predict '<json>'` | Predice un registro propio, validándolo con el contrato |
| `python -m pytest -v` | Ejecuta las pruebas |
| `mlflow ui --backend-store-uri sqlite:///mlflow.db` | Abre la interfaz de MLflow |

En Linux/macOS existen atajos equivalentes: `make all`, `make validate`, `make train`,
`make evaluate`, `make predict`, `make test`, `make clean` (`make help` los lista).

Ejemplo de inferencia con un registro propio:

```bash
python -m src.predict '{"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.6, "petal_width": 2.2}'
```

> En Windows PowerShell las comillas dentro del JSON pueden llegar mal a Python según la versión
> de PowerShell. Si aparece un error de JSON, haga la predicción desde el notebook (sección 6).

## 11. Resultados

| Métrica | Valor |
|---|---|
| Accuracy en prueba (30 flores) | **0.9333** |
| F1 macro en prueba | **0.9333** |
| Accuracy validación cruzada, 5 folds (`src.train`) | 0.9583 ± 0.0264 |
| Compuerta de calidad (≥ 0.90) | Superada |

Matriz de confusión en prueba (filas = especie real, columnas = predicha):

| | setosa | versicolor | virginica |
|---|---|---|---|
| **setosa** | 10 | 0 | 0 |
| **versicolor** | 0 | 9 | 1 |
| **virginica** | 0 | 1 | 9 |

*Setosa* se clasifica sin errores; las únicas confusiones son entre *versicolor* y *virginica*,
que se solapan en tamaño. La evidencia completa de ejecución está en
[`reports/pipeline_run.log`](reports/pipeline_run.log) y en el notebook.

## 12. Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `No suitable Python runtime found` al ejecutar `py -3.13 -m venv .venv` | Esa versión de Python no está instalada | Ejecute `py --list` y use una versión que aparezca (ej. `py -3.14`), o `py -m venv .venv` |
| `python` / `py` no se reconoce como comando | Python no está instalado o no está en el PATH | Instálelo (sección 3) marcando *"Add python.exe to PATH"* y reabra la terminal |
| *"La ejecución de scripts está deshabilitada"* al activar el entorno | Política de ejecución de PowerShell | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y vuelva a activar |
| `ModuleNotFoundError: No module named 'src'` | Se ejecutó `run_pipeline.py` fuera de la carpeta del proyecto o sin la carpeta `src/` al lado | Entre a la raíz del proyecto (`cd iris-pipeline`) y compruebe con `dir`/`ls` que están `src`, `contracts`, `data` y `tests` |
| `ModuleNotFoundError: No module named 'pandas'` (u otra librería) | El entorno virtual no está activo o faltan dependencias | Active el entorno (debe verse `(.venv)`) y ejecute `pip install -r requirements.txt` |
| `MLflow no está instalado` | Igual que el anterior | `pip install -r requirements.txt` con el entorno activo |
| `mlflow` no se reconoce como comando | El entorno virtual no está activo | Actívelo (paso 3) |
| Falla la instalación de una dependencia | pip desactualizado o versión de Python muy nueva | `python -m pip install --upgrade pip`; si persiste, cree el entorno con Python 3.12 |
| `make: command not found` (Windows) | Windows no trae `make` | Use `python run_pipeline.py`, que hace lo mismo |
| `[X] PIPELINE DETENIDO: ... problemas de calidad de datos` | Los datos de entrada no cumplen el contrato | Revise las líneas `ERROR:` del log; corrija el CSV o el contrato |
| `PermissionError` al escribir en `reports/` o `models/` | La carpeta está sincronizando con OneDrive o un archivo está abierto | Cierre los archivos abiertos o pause la sincronización y vuelva a ejecutar |

## 13. Decisiones de reproducibilidad

- **Semilla fija** (`random_state=42`): la partición y el modelo son idénticos en cada ejecución;
  una prueba automatizada lo verifica.
- **Partición estratificada**: train y test conservan 1/3 de cada especie.
- **Sin fuga de información**: el escalado está dentro del `Pipeline`, así que solo aprende de
  los datos de entrenamiento; la etapa 2 comprueba que ningún registro esté en ambas partes.
- **Configuración centralizada** en `src/config.py`: ninguna ruta ni parámetro está repetido.
- **Datos crudos inmutables**: `data/raw/` nunca se modifica; todo lo derivado se regenera.
- **Artefactos no versionados**: modelos, métricas, datos procesados y MLflow están en
  `.gitignore` porque se reproducen con un comando. Solo se versiona el log como evidencia.
- **Dependencias declaradas** en `requirements.txt`; el entorno virtual (`.venv/`) no se sube.
- **Integración continua**: GitHub Actions comprueba en cada *push* que el pipeline corre desde
  cero en una máquina limpia.

---

*Referencia del dataset:* Fisher, R. A. (1936). The use of multiple measurements in taxonomic
problems. *Annals of Eugenics, 7*(2), 179–188.
