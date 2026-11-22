# Clasificación supervisada: Naive Bayes, SVM y red neuronal

Proyecto final de Ciencia de Datos. Se predice si un cliente de un banco contratará un depósito a plazo tras una campaña telefónica, usando el dataset [Bank Marketing (UCI)](https://archive.ics.uci.edu/dataset/222/bank-marketing). Se comparan tres modelos (Naive Bayes, SVM y perceptrón multicapa) contra una regresión logística como modelo de referencia.

## Estructura

```text
├── data/
│   ├── raw/bank-full.csv        # dataset original (45,211 filas, 17 columnas)
│   └── processed/               # datos preprocesados y preprocesador ajustado
├── notebooks/                   # EDA (01_eda_bank_marketing.ipynb)
├── src/
│   ├── config.py                # rutas, semilla y parámetros
│   ├── data_prep.py             # limpieza, split y preprocesamiento
│   ├── models.py                # modelos y rejillas de hiperparámetros
│   ├── train.py                 # búsqueda con validación cruzada
│   └── evaluate.py              # métricas, curvas y análisis de errores
├── outputs/
│   ├── figures/                 # gráficas del EDA, curvas ROC y PR
│   └── metrics/                 # resultados y modelos entrenados
├── paper/                       # reporte en formato IEEE (PDF)
└── requirements.txt
```

## Cómo ejecutarlo

Requiere Python 3.10, 3.11 o 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data_prep          # preprocesa y guarda los datos
python -m src.train              # busca hiperparámetros y entrena los 4 modelos
python -m src.evaluate           # métricas en prueba y curvas ROC / PR
python -m src.error_analysis     # errores por grupo (job, month, poutcome...)
```

El entrenamiento del SVM es la parte más lenta (varios minutos).

El EDA está en `notebooks/01_eda_bank_marketing.ipynb`.

## Reproducibilidad

| Parámetro | Valor |
|---|---|
| Semilla (`RANDOM_STATE`) | 42 |
| Proporción de prueba (`TEST_SIZE`) | 0.20, estratificado |
| Folds de validación cruzada (`CV_SPLITS`) | 5, estratificados |
| Métrica de selección | F1 |

El escalado, la imputación y la codificación se ajustan solo con el conjunto de entrenamiento. Todos los parámetros están en `src/config.py`.

## Decisiones principales

- **Desbalance:** solo 11.7 % de los clientes contrata el depósito, así que la exactitud no basta. Se reportan precisión, recall, F1, ROC-AUC y PR-AUC.
- **`pdays`:** el valor `-1` (sin contacto previo) se convierte en faltante y se agrega el indicador `was_contacted_prev`.
- **`duration`:** se conserva (`DROP_DURATION = False`), aunque solo se conoce después de la llamada. Se puede excluir cambiando esa bandera.
- **SVM:** por costo computacional, la búsqueda usa un submuestreo estratificado de 10,000 filas del entrenamiento.

## Resultados (conjunto de prueba, 9,043 filas)

| Modelo | F1 (CV, media ± std) | Exactitud | Precisión | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Regresión logística (referencia) | 0.550 ± 0.009 | 0.846 | 0.418 | 0.813 | 0.552 | 0.908 | 0.537 |
| Naive Bayes | 0.437 ± 0.008 | 0.847 | 0.388 | 0.526 | 0.446 | 0.807 | 0.444 |
| SVM | 0.477 ± 0.043 | 0.900 | 0.596 | 0.448 | 0.511 | 0.889 | 0.529 |
| Red neuronal (MLP) | 0.554 ± 0.023 | 0.907 | 0.648 | 0.455 | 0.534 | 0.928 | 0.612 |

Mejores hiperparámetros, espacio de búsqueda y tiempos: `outputs/metrics/cv_training_summary.json`. Resultados completos y matrices de confusión: `outputs/metrics/evaluation_results.csv` y `error_analysis.json`. Análisis de errores por grupo: `outputs/metrics/error_by_group.csv` y `outputs/figures/error_recall_by_group.png`.