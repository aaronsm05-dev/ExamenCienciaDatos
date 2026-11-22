# Clasificación con redes neuronales, clasificador bayesiano y SVM
Aplicando procesos de ciencia de datos para predecir si en una campaña de marketing de uns institución bancaria, dadas ciertas variables el cliente se va a suscribir a un depósito a plazo.

El proyecto abarca desde la limpieza, exploración y visualización de las variables hasta el entrenamiento, evaluación e interpretación de modelos de machine learning. Todo el entorno y las dependencias están gestionados de forma reproducible mediante entorno virtual y control de versiones en Git.

## Estructura en arbol del proyecto

```text
ExamenCienciaDatos/
├── data/               # Datos del dataset (crudos y procesados)
├── notebooks/          # Notebooks de exploración EDA
├── src/                # Scripts de código
├── paper/              # Se guarda el reposrte IEEE
├── models/             # Modelos entrenados guardados
├── outputs/            # Gráficas y resultados generados, incluyendo los modelos y las evaluaciones
├── .gitignore          # Archivos y carpetas ignorados por Git
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación general del proyecto
```

## Requisitos y entorno 

Se recomienda utilizar **Python 3.10, 3.11 o 3.12** (compatible con los rangos de versiones definidos en `requirements.txt`).

### Crear y activar el entorno virtual (`.venv`) e instalar dependencias

**En Windows (PowerShell / CMD):**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Librerías del proyecto (requirements.txt)

- **numpy**: arreglos numéricos multidimensionales y operaciones matemáticas eficientes.   
- **pandas**: manipulación, limpieza y análisis de datos tabulares (DataFrame).   
- **scipy**: funciones científicas, estadísticas y álgebra numérica.   
- **scikit-learn**: preprocesamiento, división de datos, modelos de machine learning (redes neuronales, clasificadores bayesianos, SVM, etc.) y métricas de evaluación.   
- **matplotlib**: creación de gráficas estáticas y visualizaciones base.   
- **seaborn**: visualizaciones estadísticas de alto nivel basadas en matplotlib.   
- **plotly**: gráficos interactivos para exploración de datos y resultados.   
- **xgboost**: implementación de gradient boosting optimizada sobre árboles de decisión.   
- **lightgbm**: algoritmo de gradient boosting rápido y ligero basado en árboles.   
- **shap**: interpretación y explicabilidad de predicciones de modelos.   
- **joblib**: guardado y carga eficiente de modelos entrenados y objetos en disco.   
- **jupyterlab**: entorno interactivo basado en navegador para trabajar con notebooks.   
- **ipykernel**: kernel de Python para ejecutar celdas de notebooks en Jupyter y editores.   
- **streamlit**: creación rápida de aplicaciones web interactivas para mostrar o probar resultados.   
- **python-dotenv**: carga de variables de entorno desde archivos .env.   
- **openpyxl**: lectura y escritura de archivos de Excel (.xlsx).

## Análisis Exploratorio de Datos (EDA)

En esta etapa se cargó, inspeccionó y limpió el conjunto de datos para comprender su estructura, calidad y distribución antes de pasar al modelado.

El dataset original cuenta con **45,211 registros** y **17 columnas** (`7` numéricas de tipo `int64` y `10` categóricas de tipo `object`)

### Limpieza y calidad de datos
**Valores nulos y duplicados:** No se encontraron valores nulos explícitos (`isna().sum() == 0`) ni filas duplicadas (`duplicated().sum() == 0`).
**Etiqueta `'unknown'` en variables categóricas:** Aunque no hay nulos tradicionales, varias columnas utilizan `'unknown'` para representar información faltante o no registrada:
  * `job`: 288 registros (**0.64%**)
  * `education`: 1,857 registros (**4.11%**)
  * `contact`: 13,020 registros (**28.80%**)
  * `poutcome`: 36,959 registros (**81.75%**)
**Transformación del objetivo (`y`):** La variable de salida se convirtió de texto a formato binario (`'yes'` $\rightarrow$ `1`, `'no'` $\rightarrow$ `0`) 

### Análisis univariado y bivariado
Para comprender y entender las variables de forma individual y también relacionadas entre sí.

**Desbalance de clases (`y`):** Existe un marcado desbalanceo hacia la clase negativa:
  * Clase `0` : **~88.30%** (~39,922 casos).
  * Clase `1` : **~11.70%** (~5,289 casos).

Todas las gráficas resultantes del análisis estan en la carpeta de figures dentro de la carpeta outputs


### Conclusiones del EDA

1. **El desbalanceo condiciona el modelado y las métricas:** Con solo **11.70%** de casos positivos, un modelo trivial podría alcanzar ~88.3% de *accuracy* prediciendo siempre `0`. Será clave usar métricas como **F1-score, Recall, Precision y ROC-AUC**, además de evaluar estrategias como ponderación de clases (`class_weight`) o ajuste de umbral.
2. **Distribuciones sesgadas y valores atípicos:** Variables como `balance`, `duration`, `campaign` y `previous` presentan colas largas hacia la derecha y valores extremos. Convendrá revisar transformaciones, escalado robusto o modelos basados en árboles, así como analizar si `duration` debe tratarse con cuidado según el alcance real de predicción antes de la llamada.
3. **Tratamiento específico de `'unknown'` en variables como `poutcome` tiene un **81.75%** y `contact` un **28.80%**, lo cual puede reflejar comportamiento real del proceso (por ejemplo, clientes nuevos sin campaña previa) más que simple ruido. Asimismo, el -1 en `pdays` actúa como valor centinela (sin contacto previo) y no debe interpretarse literalmente como una distancia temporal continua sin tratamiento previo.

## Código fuente (src)

El directorio `src/` reúne la lógica reutilizable del proyecto en scripts de Python, separando la configuración, el procesamiento de datos, el entrenamiento y la evaluación para mantener el código ordenado y fácil de mantener.

### Configuración (`src/config.py`)

Centraliza las rutas del proyecto, los parámetros de reproducibilidad y las opciones de experimentación para no tener valores fijos repartidos en distintos scripts[cite: 16].

* **Ruta base del proyecto (`BASE_DIR`):** Calcula de forma dinámica la carpeta raíz del repositorio a partir de la ubicación de `config.py` usando `pathlib.Path`, evitando rutas absolutas que fallen en otras computadoras o entornos.
* **Rutas de entrada y salida:** Define dónde se lee el dataset original (`DATA_RAW_PATH` apuntando a `data/raw/bank-full.csv`) y dónde se guardan los resultados generados (`DATA_PROCESSED_DIR`, `OUTPUTS_DIR`, `FIGURES_DIR`, `METRICS_DIR` y `PAPER_DIR`).
* **Reproducibilidad y validación (`RANDOM_STATE`, `TEST_SIZE`, `CV_SPLITS`):** Establece la semilla aleatoria (`42`), la proporción del conjunto de prueba (`0.20`, es decir, 20%) y el número de particiones para validación cruzada (`5`) para que los experimentos sean consistentes y comparables.
* **Control de la variable `duration` (`DROP_DURATION`):** Bandera booleana (`False` por defecto) pensada para activar o desactivar fácilmente la exclusión de `duration` al probar modelos o analizar su efecto.
* **Creación automática de carpetas:** Al importar el módulo, recorre las carpetas de salida (`DATA_PROCESSED_DIR`, `FIGURES_DIR`, `METRICS_DIR` y `PAPER_DIR`) y las crea si aún no existen (`mkdir(parents=True, exist_ok=True)`), previniendo errores por directorios faltantes.

### Preparación de datos (`src/data_prep.py`)

Se encarga de cargar el dataset crudo, limpiar valores especiales, dividir los datos sin fuga de información (*data leakage*), aplicar transformaciones numéricas y categóricas con `scikit-learn`, y opcionalmente guardar los conjuntos procesados y el preprocesador en disco.

* **Indicador de contacto previo (`add_pdays_indicator`):** Trabaja sobre una copia del DataFrame (`.copy()`) para no mutar el original. Si existe `pdays`, crea `was_contacted_prev` (`1` si `pdays != -1`, `0` si fue `-1`) convirtiendo el booleano a entero con `.astype(int)`, y reemplaza `-1` por `np.nan` para que deje de tratarse como número real.
* **Carga y división estratificada (`load_and_split_data`):** Lee el CSV con `sep=";"`, aplica `add_pdays_indicator` y elimina `duration` si `drop_duration` está activo. Mapea `y` de `'no'`/`'yes'` a `0`/`1`, separa `X` y `y`, y usa `train_test_split` con `stratify=y` para mantener la misma proporción de clases en *train* y *test*.
* **Preprocesador con `ColumnTransformer` (`build_preprocessor`):** Detecta columnas numéricas (`include=[np.number]`) y categóricas (`exclude=[np.number]`). Arma dos `Pipeline`s:
  * **Numérico:** imputa faltantes con la mediana (`SimpleImputer(strategy="median")`) y escala con media `0` y desviación estándar `1` (`StandardScaler`).
  * **Categórico:** imputa faltantes con la moda (`strategy="most_frequent"`) y codifica con `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` para evitar fallos ante categorías nuevas y devolver un arreglo denso compatible con modelos que no aceptan matrices dispersas.
  * Une ambos con `ColumnTransformer(..., remainder="drop")`, descartando cualquier columna no contemplada en esas listas.
* **Orquestación y guardado (`prepare_datasets`):** Obtiene el *split* crudo, ajusta (`fit_transform`) el preprocesador **solo en `X_train_raw`** y transforma (`transform`) `X_test_raw` para evitar *data leakage*. Extrae los nombres finales de variables con `preprocessor.get_feature_names_out()` y, si `save_processed=True`, guarda los arreglos comprimidos en `bank_marketing_prepared.npz` (`np.savez_compressed`) y el objeto ajustado en `preprocessor.joblib` (`joblib.dump`).
* **Ejecución directa (`if __name__ == "__main__"`):** Permite correr el script como programa principal para generar y guardar los datos procesados de una vez e imprimir las dimensiones finales de *train*, *test* y el total de *features*

### Modelos y sus hiperparámetros (`src/models.py`)

Define y centraliza los estimadores de clasificación y sus respectivas cuadrículas de hiperparámetros (`param_grid`) para la búsqueda y evaluación de modelos sobre el mismo conjunto de configuraciones.

#### Teoría detrás de la configuración de los modelos

* **Regresión logística (`logistic_regression`):** Es un modelo lineal clásico y rápido que sirve como buena referencia inicial para ver qué tanto explica una relación sencilla entre variables. Se subió `max_iter=1000` para darle suficiente margen de pasos y que logre converger sin quejarse. Variar `C` controla qué tan estricto es el freno contra el sobreajuste (valores chicos regularizan más; valores grandes dejan que el modelo se ajuste más a los datos), y probar `class_weight` (`None` vs `"balanced"`) permite ver si darle más peso a la clase minoritaria ayuda a detectar mejor los casos positivos.
* **Naive Bayes Gaussiano (`naive_bayes`):** Asume que las variables numéricas siguen una distribución parecida a una campana (normal) y que aportan información de forma independiente entre sí. Como es un modelo muy rápido y sencillo, `var_smoothing` sirve como un "colchón" de seguridad para evitar divisiones entre cero cuando una variable tiene varianza muy baja o casi nula en algún grupo.
* **Máquina de vectores de soporte (`svm`):** Busca la frontera que mejor separa las clases dejando el mayor margen posible entre ellas. Se probó el kernel `linear` (frontera recta) y `rbf` (frontera curva para relaciones más complejas), jugando con `C` para equilibrar entre un margen amplio y cometer pocos errores en entrenamiento. Se activó `probability=True` porque permite estimar probabilidades (útil para curvas ROC y ajustar umbrales), pero como eso vuelve el entrenamiento más pesado en datasets grandes, se dejaron pocas combinaciones y `cache_size=512` para hacerlo viable en tiempo razonable.
* **Perceptrón multicapa (`mlp`):** Es una red neuronal artificial capaz de aprender patrones no lineales combinando varias capas de neuronas. La activación `relu` ayuda a que la red aprenda relaciones no lineales sin frenarse tanto como con activaciones antiguas, y el optimizador `adam` ajusta los pesos de forma adaptativa. Probar distintos tamaños de capas ocultas (`(64, 32)` y `(128, 64)`), tasas de aprendizaje (`learning_rate_init`) y regularización (`alpha`) busca encontrar el punto medio entre aprender lo suficiente y no memorizar el ruido, apoyado por `early_stopping=True` para cortar el entrenamiento si ya no mejora en validación.

#### Explicación del código

* **Función `get_models_and_grids`:** Recibe `random_state` (por defecto `RANDOM_STATE` desde `src/config.py`) y devuelve un diccionario donde cada clave identifica un modelo y contiene su instancia (`estimator`) junto con los hiperparámetros a probar (`param_grid`).

* **Regresión logística (`logistic_regression`):** Instancia `LogisticRegression(max_iter=1000, random_state=random_state)` y prueba valores de regularización `C` (`[0.1, 1.0, 10.0]`) junto con `class_weight` (`[None, "balanced"]`) para evaluar el efecto de compensar el desbalanceo de clases.

* **Naive Bayes Gaussiano (`naive_bayes`):** Instancia `GaussianNB()` y ajusta `var_smoothing` (`[1e-9, 1e-8, 1e-7]`), un parámetro de estabilidad numérica que suma una fracción de la mayor varianza a las varianzas de cada característica para evitar divisiones por cero o varianzas nulas.

* **Máquina de vectores de soporte (`svm`):** Configura `SVC(probability=True, cache_size=512, random_state=random_state)` y define `param_grid` como una **lista de diccionarios** (sintaxis de `scikit-learn` para evaluar combinaciones de hiperparámetros distintas por *kernel*, evitando probar `gamma` en el kernel `linear` donde no aplica).
  * *Por qué quedó de esta forma:* Entrenar `SVC` con `probability=True` sobre un dataset grande (~45 mil registros en total y validación cruzada) es computacionalmente costoso, por lo que se acotó la búsqueda a `kernel` en `["linear", "rbf"]`, valores reducidos de `C` (`[0.1, 1.0]`), `gamma: ["scale"]` únicamente en `rbf` y `cache_size=512` (MB de caché) para acelerar el entrenamiento sin saturar la memoria.

* **Perceptrón multicapa (`mlp`):** Usa `MLPClassifier` con activación `relu`, optimizador `adam`, `max_iter=300` y parada temprana (`early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=15`) para detener el entrenamiento si deja de mejorar en validación y evitar sobreajuste[cite: 18]. Su `param_grid` explora arquitecturas `hidden_layer_sizes` (`(64, 32)` y `(128, 64)`), regularización L2 `alpha` (`[1e-4, 1e-3, 1e-2]`) y tasa de aprendizaje inicial `learning_rate_init` (`[1e-3, 5e-4]`).

### Entrenamiento y búsqueda de hiperparámetros (`src/train.py`)

Se encarga de cargar los datos preparados, buscar los mejores hiperparámetros con validación cruzada estratificada, guardar el mejor estimador de cada modelo y exportar un resumen de resultados junto con el conjunto de prueba.

* **Carga de datos (`load_training_data`):** Revisa si ya existe `bank_marketing_prepared.npz` en `DATA_PROCESSED_DIR`; si está ahí, carga los arreglos con `np.load(..., allow_pickle=True)`, y si no, llama a `prepare_datasets(save_processed=True)` para generarlos y guardarlos primero.

* **Función `train_all_models`:** Coordina todo el ciclo de entrenamiento y ajuste:
  * **Preparación:** Carga los datos y obtiene las configuraciones desde `get_models_and_grids(random_state=random_state)`. Si se pasa `selected_models`, filtra el diccionario para entrenar únicamente los modelos indicados.
  * **Estrategia de validación cruzada (`StratifiedKFold`):** Crea `cv_splits` particiones estratificadas con `shuffle=True` para mantener la proporción de clases en cada *fold* y asegurar que los modelos se evalúen en las mismas divisiones.
  * **Bucle de entrenamiento y ajuste:** Por cada modelo, mide el tiempo con `time.time()` y configura `GridSearchCV` con la métrica `scoring` (por defecto `"f1"`), `refit=True` (para reentrenar el mejor conjunto de hiperparámetros sobre los datos usados en la búsqueda) y `return_train_score=True` (para registrar también el desempeño en entrenamiento).
  * **Guardado del mejor modelo y métricas:** Extrae el promedio y la desviación estándar del mejor resultado en validación cruzada usando `grid_search.best_index_` sobre `cv_results_`, guarda `grid_search.best_estimator_` en `outputs/metrics/trained_models/{model_name}_best.joblib` con `joblib.dump`, y arma un diccionario con hiperparámetros, tiempo, cantidad de muestras usadas y ruta del archivo.
  * **Exportación final:** Guarda `X_test`, `y_test` y `feature_names` en `test_split.npz` para la etapa de evaluación, y escribe el resumen completo en `outputs/metrics/cv_training_summary.json`.

* **Ejecución directa (`if __name__ == "__main__"`):** Ejecuta `train_all_models()` con los valores por defecto al correr el script directamente.

> **Nota sobre SVM (`svm`):** Entrenar `SVC` con `probability=True` y validación cruzada sobre todo el conjunto de entrenamiento resulta muy lento y pesado. Por ello, cuando `model_name == "svm"`, el script fija `n_jobs_cv = 1` y, si `X_train` supera `svm_max_samples` (`10,000` por defecto), toma un submuestreo estratificado (`train_size=svm_max_samples`, `stratify=y_train`) para que el ajuste termine en un tiempo razonable.

### Evaluación de modelos (`src/evaluate.py`)

Se encarga de probar los modelos ya entrenados sobre el conjunto de prueba (`test_split.npz`) para medir qué tan bien generalizan, analizar los tipos de error y generar tablas y gráficas comparativas en disco.

* **Puntuaciones de decisión y probabilidades (`predict_proba` / `decision_function`):** Las predicciones finales (`predict`) cortan en un umbral fijo, pero evaluar curvas y áreas requiere un puntaje continuo de confianza para la clase positiva (`1`). Usar `predict_proba` (tomando la columna `[:, 1]`) entrega probabilidades calibradas; si un modelo no lo soporta, se recurre a `decision_function` (distancia al margen de decisión) o, en último caso, a la clase predicha convertida a `float`.
* **Métricas de clasificación (`accuracy`, `precision`, `recall`, `f1`):**
  * **`accuracy`:** Porcentaje total de aciertos; es engañoso en datasets desbalanceados como este, pero sirve como referencia general.
  * **`precision`:** De todos los clientes que el modelo marcó como positivos (`1`), cuántos realmente aceptaron; mide qué tanto se evitan falsas alarmas. `zero_division=0` evita advertencias o errores si algún modelo no predice ningún positivo.
  * **`recall`:** De todos los clientes que sí aceptaron en la realidad, cuántos logró detectar el modelo; clave cuando importa no dejar escapar oportunidades positivas.
  * **`f1`:** Media armónica entre `precision` y `recall`; resume en un solo número el balance entre ambos cuando las clases están desbalanceadas.
* **Métricas basadas en curvas (`roc_auc` y `pr_auc`):**
  * **`roc_auc`:** Resume la capacidad del modelo para ordenar positivos por encima de negativos a través de distintos umbrales, comparando tasa de verdaderos positivos (`TPR` / `recall`) contra tasa de falsos positivos (`FPR`).
  * **`pr_auc`:** Área bajo la curva `precision` vs `recall` (`auc(r_curve, p_curve)`); resulta especialmente informativa en problemas desbalanceados porque se enfoca en el desempeño sobre la clase minoritaria sin inflarse por los numerosos verdaderos negativos.
* **Matriz de confusión y análisis de errores (`tn`, `fp`, `fn`, `tp`):** Desglosa exactamente dónde acierta y dónde falla cada modelo. Separar falsos positivos (`fp`, contactar a quien no contratará) de falsos negativos (`fn`, perder a un cliente potencial) permite entender el costo práctico de cada error, junto con el total de fallos y la tasa de error sobre `len(y_test)`.
* **Gráficas y archivos de salida (`roc_curves.png`, `pr_curves.png`, `evaluation_results.csv`, `error_analysis.json`):** Guardar las curvas a `dpi=300` con `plt.tight_layout()` asegura imágenes nítidas y sin recortes de etiquetas para reportes o presentaciones, mientras que el CSV y el JSON dejan las métricas y el desglose de errores listos para consultarse o citarse sin tener que reejecutar los modelos.

## Ejecución en terminal

Desde la raíz del proyecto (con tu entorno virtual activado), ejecuta los módulos en orden para preparar los datos, entrenar los modelos y evaluar resultados:

```bash
python -m src.data_prep
python -m src.train
python -m src.evaluate
```
