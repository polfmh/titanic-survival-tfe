# Bloque de Machine Learning aplicado al Titanic
### Trabajo de Fin de Estudios — Posgrado en Business Analytics (PBA)
### UPC School, 3ª Edición · Autor: Albert

---

## 1. Introducción y Objetivo del Bloque

Este bloque constituye el núcleo técnico del proyecto y responde directamente a la pregunta de investigación del TFE:

> *¿Qué factores determinan la probabilidad de supervivencia en un desastre marítimo y cómo pueden utilizarse para mejorar la gestión de evacuaciones?*

Para abordarla, se ha trabajado con el dataset histórico del Titanic, proveniente de la competición **Titanic — Machine Learning from Disaster** de Kaggle (una de las competiciones de iniciación más utilizadas en el mundo de la ciencia de datos, con más de 16.000 equipos participantes). El objetivo no es únicamente obtener una puntuación alta en el ranking de la competición, sino construir un modelo interpretable capaz de cuantificar la influencia de variables sociodemográficas en la supervivencia y generar probabilidades individuales que permitan un análisis profundo y comparativo.

Los objetivos específicos del bloque son:

1. Explorar y comprender los patrones del dataset Titanic.
2. Aplicar técnicas de limpieza e ingeniería de variables para maximizar la calidad del análisis.
3. Construir y evaluar modelos de clasificación supervisada.
4. Explicar las predicciones mediante técnicas de interpretabilidad (SHAP values).
5. Generar un dataset enriquecido con probabilidades de supervivencia reutilizable por el resto del equipo.
6. Producir el archivo de predicciones para la competición de Kaggle.

---

## 2. Fuente de Datos y Descripción del Dataset

### 2.1 Fuente

El dataset proviene de la competición oficial de Kaggle:
**https://www.kaggle.com/competitions/titanic**

Se trata de datos históricos recopilados de los registros de pasajeros del RMS Titanic, el trasatlántico que se hundió el 15 de abril de 1912 tras colisionar con un iceberg en su viaje inaugural de Southampton a Nueva York. De los aproximadamente 2.224 pasajeros y tripulantes a bordo, más de 1.500 perdieron la vida.

### 2.2 Estructura del Dataset

El dataset se divide en dos archivos CSV:

| Archivo | Filas | Descripción |
|---------|-------|-------------|
| `train.csv` | 891 | Pasajeros con variable objetivo (`Survived`) conocida |
| `test.csv` | 418 | Pasajeros sin variable objetivo (para el submission de Kaggle) |
| `gender_submission.csv` | 418 | Ejemplo del formato de entrega esperado |

El conjunto total de 1.309 pasajeros representa aproximadamente el 59% del total de personas a bordo. El dataset cubre exclusivamente pasajeros, no tripulación.

### 2.3 Diccionario de Variables

| Variable | Tipo | Descripción | Notas |
|----------|------|-------------|-------|
| `PassengerId` | int | Identificador único | No predictivo |
| `Survived` | int (0/1) | **Variable objetivo** | 0 = No sobrevivió, 1 = Sobrevivió |
| `Pclass` | int (1/2/3) | Clase del billete | Proxy del nivel socioeconómico |
| `Name` | str | Nombre completo | Contiene título social (Mr., Mrs., etc.) |
| `Sex` | str | Género | `male` / `female` |
| `Age` | float | Edad en años | ~20% de valores nulos |
| `SibSp` | int | Nº de hermanos/cónyuge a bordo | |
| `Parch` | int | Nº de padres/hijos a bordo | |
| `Ticket` | str | Número de billete | Compartido entre grupos |
| `Fare` | float | Precio del billete (£) | 1 valor nulo en test |
| `Cabin` | str | Número de cabina | ~77% de valores nulos |
| `Embarked` | str | Puerto de embarque | C=Cherbourg, Q=Queenstown, S=Southampton |

### 2.4 Análisis de Valores Nulos

| Variable | Nulos en Train | % Nulos | Estrategia |
|----------|---------------|---------|------------|
| `Age` | 177 | 19,9% | Imputación por mediana (Título + Pclass) |
| `Cabin` | 687 | 77,1% | Variable binaria `HasCabin` |
| `Embarked` | 2 | 0,2% | Moda (Southampton = 'S') |
| `Fare` | 1 (solo en test) | 0,1% | Mediana por Pclass |

La columna `Cabin` presenta un porcentaje de nulos tan elevado que su imputación directa no es viable. Se ha optado por transformarla en una variable indicadora (`HasCabin`), que captura la información relevante: si un pasajero tiene o no cabina registrada correlaciona con su clase social y, por tanto, con su ubicación en el barco.

### 2.5 Limitaciones del Dataset

- El dataset cubre únicamente pasajeros, no tripulación.
- Los datos son históricos y pueden contener errores de registro propios de 1912.
- El 77% de los valores de cabina son nulos, limitando el análisis de ubicación física en el barco.
- La muestra de 891 pasajeros es relativamente pequeña para modelos de alta complejidad, lo que aumenta el riesgo de sobreajuste.
- La variable objetivo presenta desequilibrio de clases: 61,6% no sobrevivieron vs. 38,4% sobrevivieron.

---

## 3. Análisis Exploratorio de Datos (EDA)

El análisis exploratorio tiene como objetivo identificar patrones, distribuciones y relaciones entre variables antes de construir el modelo.

### 3.1 Distribución de la Variable Objetivo

De los 891 pasajeros del conjunto de entrenamiento:
- **549 no sobrevivieron** (61,6%)
- **342 sobrevivieron** (38,4%)

El dataset presenta un cierto desequilibrio de clases que debe tenerse en cuenta durante la evaluación del modelo. Un clasificador que predijera "nadie sobrevive" obtendría ya un 61,6% de accuracy, por lo que se han utilizado métricas adicionales como el ROC-AUC para una evaluación más completa.

### 3.2 Supervivencia por Género

El género es la variable más determinante del dataset:

| Género | Total | Supervivientes | Tasa de supervivencia |
|--------|-------|---------------|----------------------|
| Mujer | 314 | 233 | **74,2%** |
| Hombre | 577 | 109 | **18,9%** |

Las mujeres sobrevivieron a una tasa casi 4 veces superior a la de los hombres. Este patrón refleja el protocolo de evacuación "mujeres y niños primero" aplicado durante el desastre, así como la norma social de la época.

### 3.3 Supervivencia por Clase Social (Pclass)

| Clase | Total | Supervivientes | Tasa |
|-------|-------|---------------|------|
| 1ª clase | 216 | 136 | **63,0%** |
| 2ª clase | 184 | 87 | **47,3%** |
| 3ª clase | 491 | 119 | **24,2%** |

Los pasajeros de primera clase mostraron tasas de supervivencia significativamente superiores. Esto se explica por dos factores acumulativos: (1) su posición socioeconómica les otorgó mayor influencia durante la evacuación, y (2) sus camarotes estaban ubicados en cubiertas superiores, más cercanas a los botes salvavidas.

### 3.4 Interacción Género × Clase

La combinación de género y clase produce los patrones más extremos del dataset:

| | 1ª Clase | 2ª Clase | 3ª Clase |
|---|---------|---------|---------|
| **Mujer** | 96,8% | 92,1% | 50,0% |
| **Hombre** | 36,9% | 15,7% | 13,5% |

Una mujer de primera clase tenía un 96,8% de probabilidad de sobrevivir, mientras que un hombre de tercera clase tenía solo un 13,5%. La diferencia de 83,3 puntos porcentuales entre estos dos perfiles extremos ilustra cómo la intersección de género y clase social determinó en gran medida el acceso a los botes salvavidas.

### 3.5 Supervivencia por Edad

El análisis por grupos de edad muestra un patrón más complejo:
- Los niños (0-12 años) tuvieron mayor tasa de supervivencia que el promedio, confirmando el principio de "niños primero".
- Los adultos (18-60 años) se acercaron a la media global.
- Las diferencias de edad pierden significado estadístico cuando se controla por género y clase, ya que estas dos variables tienen mayor poder explicativo.

La media de edad de los supervivientes (28,3 años) fue ligeramente inferior a la de los no supervivientes (30,6 años), pero la diferencia no es estadísticamente concluyente sin considerar otras variables.

### 3.6 Supervivencia por Tamaño Familiar

Los pasajeros que viajaban solos mostraron tasas de supervivencia inferiores a los que viajaban con familias pequeñas (2-4 miembros). Sin embargo, las familias grandes (5 o más miembros) también mostraron tasas más bajas, probablemente porque la coordinación de una evacuación en grupo numeroso resultaba más difícil.

| Perfil | Tasa de supervivencia |
|--------|----------------------|
| Solo | ~30,4% |
| Familia pequeña (2-4) | ~57,8% |
| Familia grande (5+) | ~16,1% |

---

## 4. Preprocesamiento y Feature Engineering

### 4.1 Estrategia General

Se ha combinado el conjunto de entrenamiento y el de test durante las transformaciones para garantizar que los estadísticos utilizados (medianas, modas) sean consistentes entre ambos conjuntos. Al finalizar, se vuelven a separar.

### 4.2 Extracción del Título Social

El nombre de cada pasajero contiene un título honorífico que aporta información relevante sobre género, edad aproximada y posición social. Se ha extraído mediante expresión regular y agrupado en cinco categorías:

| Título | Descripción | Ejemplos originales |
|--------|-------------|---------------------|
| `Mr` | Hombre adulto | Mr. |
| `Mrs` | Mujer casada | Mrs., Mme. |
| `Miss` | Mujer soltera/niña | Miss., Ms., Mlle. |
| `Master` | Niño varón (≤14 años) | Master. |
| `Rare` | Títulos poco comunes | Dr., Rev., Col., Countess., etc. |

El título `Master` es especialmente valioso: era el honorífico utilizado históricamente para los niños varones, por lo que actúa como proxy de la edad sin depender de los valores nulos de `Age`.

**Tasa de supervivencia por título:**
- Mrs: ~79,2% · Miss: ~70,1% · Master: ~57,5% · Rare: ~44,4% · Mr: ~15,7%

### 4.3 Imputación de la Edad

En lugar de imputar la edad con la mediana global (28 años), se ha utilizado la mediana por grupo (Título + Pclass), lo que produce estimaciones más realistas:

| | 1ª Clase | 2ª Clase | 3ª Clase |
|--|---------|---------|---------|
| **Master** | 4 | 2 | 4,5 |
| **Miss** | 30 | 20 | 18 |
| **Mr** | 40 | 30 | 26 |
| **Mrs** | 45 | 30 | 31 |
| **Rare** | 49 | 42 | — |

Esta estrategia evita asignar 28 años a un niño (Master) o a una persona mayor (Rare).

### 4.4 Variables Nuevas Creadas

| Variable | Descripción | Tipo |
|----------|-------------|------|
| `Title` | Título social extraído del nombre | Categórica (5 valores) |
| `FamilySize` | `SibSp + Parch + 1` | Numérica (1-11) |
| `IsAlone` | 1 si viaja solo (`FamilySize == 1`) | Binaria |
| `FamilyCat` | Solo / Familia pequeña / Familia grande | Categórica (3 valores) |
| `HasCabin` | 1 si tiene cabina registrada | Binaria |
| `Deck` | Letra de la cubierta (de la cabina) | Categórica (A-G + Unknown) |
| `AgeGroup` | Niño / Adolescente / Adulto joven / Adulto / Mayor | Categórica ordinal |
| `FareBand` | Cuartil de la tarifa (Bajo / Medio-bajo / Medio-alto / Alto) | Categórica ordinal |

### 4.5 Codificación Final

Para el modelo de Machine Learning, todas las variables categóricas se han codificado a variables numéricas:
- `Sex` → `Sex_num` (1=mujer, 0=hombre)
- `Embarked` → `Embarked_num` (codificación ordinal: C=0, S=1, Q=2)
- `Title` → `Title_num` (codificación ordinal por LabelEncoder)
- `AgeGroup`, `FareBand`, `FamilyCat` → codificación ordinal numérica

**Features finales del modelo (14 variables):**
`Pclass`, `Sex_num`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked_num`, `Title_num`, `HasCabin`, `FamilySize`, `IsAlone`, `AgeGroup_num`, `FareBand_num`, `FamilyCat_num`

---

## 5. Metodología de Machine Learning

### 5.1 Planteamiento del Problema

Se trata de un problema de **clasificación binaria supervisada**: dado un conjunto de características de un pasajero, predecir si sobrevivió (1) o no (0). Adicionalmente, se requiere la **probabilidad de supervivencia** (no solo la clase predicha), para permitir un análisis más rico y graduado.

### 5.2 Partición de los Datos

| Conjunto | Filas | Uso |
|----------|-------|-----|
| Entrenamiento (80%) | 712 | Ajuste del modelo |
| Validación (20%) | 179 | Evaluación interna |
| Test Kaggle | 418 | Submission final |

La partición se realizó con estratificación (`stratify=y`) para mantener la proporción original de clases en ambos conjuntos.

### 5.3 Algoritmos Evaluados

#### Regresión Logística (modelo base)
La regresión logística es un modelo lineal que estima la probabilidad de supervivencia como función logística de una combinación lineal de las variables de entrada. Se ha elegido como baseline por su interpretabilidad y rapidez. Se han normalizado las variables (`StandardScaler`) al ser un modelo sensible a la escala.

**Parámetros:** `C=1.0`, `max_iter=1000`, `solver='lbfgs'`

#### Random Forest (modelo principal)
El Random Forest es un ensemble de árboles de decisión entrenados en subconjuntos aleatorios de los datos y las variables. Combina las predicciones por votación mayoritaria, lo que reduce el sobreajuste y mejora la generalización. No requiere normalización de variables.

**Parámetros finales:** `n_estimators=200`, `max_depth=6`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features='sqrt'`

### 5.4 Estrategia de Validación

Para evaluar los modelos de forma robusta dado el tamaño reducido del dataset (891 muestras), se ha utilizado **validación cruzada estratificada de 5 folds** (`StratifiedKFold`). Esta técnica divide el conjunto en 5 partes, entrenando en 4 y evaluando en 1 de forma rotativa, garantizando que cada muestra actúe como dato de validación exactamente una vez.

### 5.5 Métricas de Evaluación

| Métrica | Descripción | Por qué se usa |
|---------|-------------|----------------|
| **Accuracy** | % de predicciones correctas | Métrica oficial de Kaggle |
| **ROC-AUC** | Área bajo la curva ROC | Evalúa la capacidad discriminatoria sin depender del umbral |
| **Precision / Recall / F1** | Métricas por clase | Útiles con desequilibrio de clases |
| **Matriz de confusión** | Distribución de TP, TN, FP, FN | Identifica el tipo de errores del modelo |

---

## 6. Resultados

### 6.1 Comparación de Modelos

| Modelo | Accuracy (val.) | ROC-AUC (val.) | CV Accuracy (5-fold) |
|--------|----------------|---------------|---------------------|
| **Regresión Logística** | **82,12%** | 85,59% | 81,03% ± 0,65% |
| **Random Forest** | 81,56% | 85,37% | **82,60% ± 2,04%** |

Ambos modelos obtienen resultados muy similares. La Regresión Logística presenta ligeramente mayor accuracy en el conjunto de validación, mientras que el Random Forest muestra mejor rendimiento promedio en la validación cruzada. El Random Forest, al ser más robusto y no lineal, se ha seleccionado como modelo principal para la generación de predicciones finales, dado que:

1. Captura relaciones no lineales e interacciones entre variables.
2. Proporciona importancia de variables directamente.
3. Es compatible con SHAP TreeExplainer, lo que permite una interpretabilidad más profunda.
4. Su mayor CV accuracy (82,60%) sugiere mejor generalización.

> **Contexto Kaggle:** El accuracy de ~82% está por encima del baseline de género (76,6%) y en el rango habitual de los mejores submissions interpretables del Leaderboard (~82-84%). Scores por encima del 85% en esta competición generalmente implican algún grado de sobreajuste al conjunto de test público.

### 6.2 Importancia de Variables (Random Forest — Gini Importance)

| Ranking | Variable | Importancia | Interpretación |
|---------|----------|-------------|----------------|
| 1 | `Sex_num` (género) | 0,2873 | **Factor más determinante** — género femenino aumenta fuertemente la supervivencia |
| 2 | `Title_num` (título) | 0,1469 | Condensa género + edad + clase social |
| 3 | `Fare` (tarifa) | 0,1267 | Proxy del nivel económico y posición en el barco |
| 4 | `Age` (edad) | 0,0795 | Efecto moderado, especialmente en niños |
| 5 | `Pclass` (clase) | 0,0762 | Acceso a botes salvavidas y posición en el barco |
| 6 | `HasCabin` | 0,0612 | Indicador de posición social y ubicación |
| 7 | `FamilySize` | 0,0452 | Tamaño del grupo de evacuación |

### 6.3 Interpretabilidad — SHAP Values

Los valores SHAP (SHapley Additive exPlanations) permiten cuantificar la contribución de cada variable a cada predicción individual, basándose en la teoría de juegos cooperativos. A diferencia de la importancia de Gini, los SHAP values tienen signo (positivo = aumenta la probabilidad de sobrevivir, negativo = la reduce).

**Principales hallazgos del análisis SHAP:**
- El género femenino tiene el mayor impacto positivo en la probabilidad de supervivencia.
- Las tarifas altas (que correlacionan con primera clase) tienen un efecto positivo consistente.
- El título `Master` (niños varones) tiene un efecto positivo, confirmando la prioridad dada a los niños.
- El género masculino y las tarifas bajas son los principales factores negativos.
- La clase `Pclass=3` tiene un efecto negativo independiente del género.

Los gráficos de SHAP generados (summary plot y bar plot) se encuentran en la carpeta `output/` y pueden utilizarse directamente en el informe y el dashboard de Power BI.

---

## 7. Outputs Generados

### 7.1 Submission para Kaggle

Archivo: `output/titanic_submission.csv`

Contiene las predicciones binarias (0/1) para los 418 pasajeros del conjunto de test, en el formato exacto requerido por la competición (columnas `PassengerId` y `Survived`).

**Resultado:** 153 supervivientes predichos (36,6%) de 418 pasajeros.

### 7.2 Dataset Enriquecido para el Equipo

Archivo: `output/titanic_enriched.csv`

Este archivo contiene los 1.309 pasajeros (891 train + 418 test) con todas las variables originales, las variables nuevas creadas y las probabilidades de supervivencia estimadas por el modelo.

| Columna | Descripción |
|---------|-------------|
| `SurvivalProb` | Probabilidad de supervivencia (0.0 – 1.0) estimada por el modelo |
| `DeathProb` | Probabilidad de fallecimiento (= 1 - SurvivalProb) |
| `ModelPred` | Predicción binaria del modelo (0 o 1) |
| `Split` | Indica si el pasajero pertenece al conjunto `train` o `test` |

Este archivo está diseñado para ser consumido directamente por Power BI sin necesidad de transformaciones adicionales.

---

## 8. Herramienta Interactiva — Simulador de Supervivencia

Se ha desarrollado una aplicación web interactiva con **Streamlit** (Python) que permite estimar la probabilidad de supervivencia de un perfil de usuario hipotético en el Titanic.

El simulador:
- Utiliza el modelo Random Forest entrenado (guardado como `model.pkl`).
- Permite al usuario seleccionar: género, edad, clase del billete, tamaño familiar y nivel de tarifa.
- Devuelve una probabilidad estimada de supervivencia con una visualización gráfica.
- Incluye una explicación contextual del perfil seleccionado.

**Ejecución:** `streamlit run simulator/app.py` desde el directorio del proyecto.

---

## 9. Conclusiones

### 9.1 Respuesta a la Pregunta de Investigación

Los datos del Titanic confirman que la probabilidad de supervivencia no fue aleatoria, sino que estuvo fuertemente determinada por factores sociodemográficos y estructurales:

1. **El género fue el factor más determinante** (importancia Gini: 28,7%). Las mujeres sobrevivieron a una tasa del 74,2%, frente al 18,9% de los hombres. El protocolo "mujeres y niños primero" es empíricamente verificable en los datos.

2. **La clase social amplifica el efecto del género.** Una mujer de primera clase tenía un 96,8% de probabilidad de sobrevivir; un hombre de tercera clase, solo un 13,5%. La clase determina tanto el estatus social como la ubicación física en el barco (cubiertas superiores vs. inferiores).

3. **El nivel económico (tarifa) tiene un efecto independiente** de la clase, probablemente porque refleja con más precisión el alojamiento exacto y el acceso a rutas de evacuación.

4. **El tamaño familiar tiene un efecto no lineal:** las familias pequeñas se beneficiaron de la cohesión grupal, mientras que los pasajeros solos y las familias grandes mostraron tasas inferiores.

5. **La edad tiene un efecto moderado**, visible principalmente en los niños varones (Master), que recibieron prioridad de evacuación.

### 9.2 Implicaciones para la Gestión de Evacuaciones

Estos resultados sugieren que en el Titanic el acceso a los medios de evacuación no fue equitativo, sino que estuvo condicionado por jerarquías sociales de la época. Para la gestión moderna de evacuaciones, el análisis identifica los perfiles más vulnerables: pasajeros en posiciones físicamente desfavorables (cubiertas inferiores), viajando solos, sin información clara de las rutas de evacuación.

### 9.3 Limitaciones del Análisis

- El modelo explica supervivencia con un 82,1% de accuracy, pero el 17,9% restante refleja variabilidad no capturada (circunstancias individuales, suerte, comportamientos específicos).
- Las variables proxy (tarifa, clase) no capturan directamente los factores causales (distancia física a los botes salvavidas).
- El dataset no incluye tripulación, lo que limita la generalización.
- El modelo se ha entrenado en datos históricos de 1912; su validación en el contexto moderno requiere la comparación con el MV Sewol (Bloque 2 del TFE).

---

## 10. Tecnologías y Herramientas Utilizadas

| Herramienta | Versión | Uso en este bloque |
|-------------|---------|-------------------|
| **Python** | 3.12.9 | Lenguaje principal |
| **pandas** | 2.2.3 | Manipulación y limpieza de datos |
| **NumPy** | — | Operaciones matriciales |
| **matplotlib** | 3.10.8 | Visualizaciones base |
| **seaborn** | 0.13.2 | Visualizaciones estadísticas |
| **scikit-learn** | 1.8.0 | Modelos ML, métricas, validación cruzada |
| **SHAP** | — | Interpretabilidad del modelo |
| **Streamlit** | — | Simulador interactivo |
| **Jupyter Notebook** | — | Entorno de desarrollo y documentación |
| **Kaggle API** | 2.1.2 | Descarga automatizada del dataset |

---

*Documento generado como parte del TFE — Posgrado en Business Analytics, UPC School · Mayo 2026*
