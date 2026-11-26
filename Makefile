# Makefile del pipeline Iris
# Uso principal (un solo comando):   make all
# En Windows sin 'make', use:         python run_pipeline.py

PYTHON ?= python3

.PHONY: help install validate train evaluate predict test all pipeline clean

help:  ## Muestra los comandos disponibles
	@echo "Comandos:"
	@echo "  make install   - instala dependencias"
	@echo "  make validate  - valida los datos de entrada"
	@echo "  make train     - entrena el modelo (valida antes)"
	@echo "  make evaluate  - valida el modelo con el conjunto de prueba"
	@echo "  make predict   - inferencia con los ejemplos del contrato"
	@echo "  make test      - ejecuta las pruebas automatizadas"
	@echo "  make all       - pipeline completo: validate -> train -> evaluate -> predict -> test"
	@echo "  make clean     - borra modelos y reportes generados"

install:
	$(PYTHON) -m pip install -r requirements.txt

validate:
	$(PYTHON) -m src.validate

train: validate
	$(PYTHON) -m src.train

evaluate: train
	$(PYTHON) -m src.evaluate

predict: evaluate
	$(PYTHON) -m src.predict

test:
	$(PYTHON) -m pytest tests -v

# Pipeline completo con un solo comando (deja log de evidencia en reports/)
all pipeline:
	$(PYTHON) run_pipeline.py

clean:
	rm -f models/*.joblib reports/*.json reports/*.log
