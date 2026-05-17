# Bloque de Comparativa: Titanic vs MV Sewol
### Trabajo de Fin de Estudios — Posgrado en Business Analytics (PBA)
### UPC School, 3ª Edición · Autor: Pablo

---

## 1. Introducción y Objetivo del Bloque

Este bloque constituye el núcleo comparativo del proyecto y responde directamente a la pregunta de investigación del TFE:

> *¿Qué factores determinan la probabilidad de supervivencia en un desastre marítimo y cómo pueden utilizarse para mejorar la gestión de evacuaciones?*

Para abordarla desde una perspectiva comparativa, se han cruzado los datos del Titanic (1912) —procesados y enriquecidos en el Bloque de Machine Learning— con los datos del MV Sewol (2014), un transbordador surcoreano que se hundió el 16 de abril de 2014 con 476 personas a bordo, la mayoría estudiantes de instituto en excursión escolar.

El objetivo no es simplemente describir los dos desastres por separado, sino identificar si los patrones de supervivencia —especialmente el de género y el de edad— se repiten o divergen entre un desastre de 1912 y uno de 2014. Si los patrones cambian, la hipótesis es que el contexto y el protocolo de evacuación explican la diferencia, por encima de las variables sociodemográficas.

Los objetivos específicos del bloque son:

1. Normalizar y unificar los datasets del Titanic y el Sewol en un formato comparable.
2. Analizar la tasa de supervivencia por género en los dos desastres.
3. Analizar la tasa de supervivencia por franja de edad en los dos desastres.
4. Cuantificar el impacto del protocolo de evacuación a través del ratio mujeres/hombres.
5. Generar un dataset combinado reutilizable para el resto del equipo.
6. Producir visualizaciones y una tabla comparativa final para la memoria y la presentación.

---

## 2. Fuentes de Datos y Descripción de los Datasets

### 2.1 Dataset del Titanic

El dataset del Titanic proviene del fichero enriquecido generado en el Bloque de Machine Learning (`output/titanic_enriched.csv`). Para este bloque se utiliza exclusivamente el conjunto de entrenamiento, que contiene los 891 pasajeros con el valor real de supervivencia conocido.

| Fichero | Filas | Descripción |
|---------|-------|-------------|
| `titanic_enriched.csv` (train) | 891 | Pasajeros con `Survived` real (0/1) |

El dataset del Titanic cubre exclusivamente pasajeros; no incluye tripulación.

### 2.2 Dataset del Sewol

El dataset del Sewol proviene del fichero `sewol/sewol_eng.csv`, que contiene 476 registros de personas a bordo del transbordador en el momento del accidente.

| Fichero | Filas | Descripción |
|---------|-------|-------------|
| `sewol_eng.csv` | 476 | Pasajeros y tripulación con información de supervivencia |

### 2.3 Diccionario de variables del Sewol

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `Category-1` | str | Tipo de persona: `sailor` (tripulación), `student`, `Normal` |
| `Category-2` | str | Subcategoría (rol específico) |
| `gender` | str | Género: `male` / `female` |
| `age` | float | Edad en años |
| `Raw` | str | Supervivencia: `survival` / `Dead` |
| `floor` | int | Planta del barco |
| `location` | str | Ubicación (front / rear) |

### 2.4 Distribución de edad del Sewol

La distribución de edad del Sewol es radicalmente diferente a la del Titanic. El 69% de los pasajeros tenían entre 10 y 20 años, lo que refleja la composición del barco: principalmente estudiantes de instituto de Danwon High School en excursión escolar.

| Franja de edad | Titanic (n) | Sewol (n) |
|----------------|------------|-----------|
| 0-12 años | 73 | 4 |
| 13-19 años | — | 327 |
| 20-39 años | — | 18 |
| 40-59 años | — | 33 |
| 60+ años | — | 29 |

Esta diferencia estructural tiene implicaciones directas en el análisis: los grupos de edad estándar del Titanic no son directamente comparables con los del Sewol sin una adaptación previa.

### 2.5 Tratamiento de valores nulos

| Variable | Dataset | Nulos | Estrategia |
|----------|---------|-------|------------|
| `Age` | Titanic | 0 (ya tratados por Albert) | — |
| `age` | Sewol | Verificado = 0 | — |

---

## 3. Metodología: Normalización y Unificación de los Datasets

### 3.1 Estrategia general

Para hacer posible la comparativa, se ha creado un DataFrame unificado (`combined`) con las columnas equivalentes de los dos desastres. La normalización consiste en traducir los dos datasets al mismo formato sin modificar los datos originales.

### 3.2 Variables normalizadas

| Columna | Titanic | Sewol | Resultado |
|---------|---------|-------|-----------|
| `disaster` | — | — | `'titanic'` / `'sewol'` |
| `gender` | `Sex` (male/female) | `gender` (male/female) | Idéntico, no hace falta transformación |
| `age` | `Age` | `age` | Idéntico, no hace falta transformación |
| `survived` | `Survived` (0/1) | `Raw` (survival/Dead → 1/0) | Mapeo texto → numérico |
| `role` | Todos `passenger` | `Category-1 == 'sailor'` → `crew`, resto → `passenger` | Variable nueva |

### 3.3 Decisión sobre la tripulación

El dataset del Titanic no incluye tripulación. Para hacer una comparativa justa, todos los análisis de supervivencia se han realizado filtrando únicamente los pasajeros (`role == 'passenger'`) de los dos datasets. Los 33 tripulantes del Sewol se han excluido del análisis principal.

| | Titanic | Sewol |
|--|---------|-------|
| Pasajeros | 891 | 443 |
| Tripulación | 0 (no incluida) | 33 |
| **Total utilizado** | **891** | **443** |

### 3.4 Dataset combinado generado

El DataFrame normalizado se ha exportado como `output/comparativa_titanic_sewol.csv` (1.334 filas, 5 columnas), listo para ser importado a Power BI o para usos posteriores.

---

## 4. Análisis Exploratorio Comparativo

### 4.1 Tasa global de supervivencia

| | Titanic | Sewol |
|--|---------|-------|
| Total pasajeros | 891 | 443 |
| Supervivientes | 342 | 149 |
| **Tasa global** | **38,4%** | **33,6%** |

Las tasas globales son similares (diferencia de 4,8 puntos porcentuales), pero esta similitud superficial esconde patrones internos muy divergentes.

### 4.2 Supervivencia por género

| Género | Titanic (n) | Tasa | Sewol (n) | Tasa |
|--------|------------|------|-----------|------|
| Mujeres | 314 | **74,2%** | 182 | **29,1%** |
| Hombres | 577 | **18,9%** | 261 | **36,8%** |
| **Ratio mujeres/hombres** | | **3,93x** | | **0,79x** |

Esta es la comparativa más relevante del TFE. En el Titanic, las mujeres sobrevivían casi 4 veces más que los hombres, reflejo directo del protocolo "mujeres y niños primero". En el Sewol, el patrón se invierte: los hombres sobrevivieron más que las mujeres (ratio 0,79x), lo que indica que el protocolo de género no se aplicó de la misma manera.

### 4.3 Supervivencia por franja de edad

Para esta comparativa se han definido grupos de edad adaptados a la distribución del Sewol:

| Franja de edad | Titanic | Sewol |
|----------------|---------|-------|
| Niño (0-12) | 57,5% | 50,0%* |
| Adolescente (13-19) | 45,3% | 23,9% |
| Adulto joven (20-39) | 33,7% | 52,2% |
| Adulto (40-59) | 40,4% | 78,2% |
| Mayor (60+) | 26,9% | 46,2% |

*\*Nota: el grupo de niños en el Sewol contiene únicamente 4 individuos. El tamaño de muestra es insuficiente para extraer conclusiones estadísticamente significativas para este grupo en este desastre.*

El patrón de edad también se invierte entre los dos desastres. En el Titanic, los grupos de menor edad (niños y adolescentes) tuvieron las tasas más altas, consistente con el protocolo "niños primero". En el Sewol, los adultos y mayores sobrevivieron mucho más que los adolescentes (que eran la mayoría a bordo). Los adolescentes del Sewol tuvieron una tasa del 23,9%, la más baja de todos los grupos.

---

## 5. Análisis del Protocolo de Evacuación

### 5.1 Dimensión de género: ratio mujeres/hombres

El ratio mujeres/hombres es el indicador que mejor captura el grado de aplicación del protocolo "mujeres primero":

| | Titanic | Sewol |
|--|---------|-------|
| Ratio mujeres/hombres | **3,93x** | **0,79x** |

En el Titanic, ser mujer multiplicaba por 3,93 la probabilidad de sobrevivir respecto a ser hombre. En el Sewol, el ratio es inferior a 1: los hombres sobrevivieron más que las mujeres. La diferencia entre los dos ratios es de 4,72 puntos, lo que sugiere que el contexto del desastre y el protocolo aplicado tuvieron un impacto determinante por encima del factor género.

### 5.2 Dimensión de edad: niños vs adultos

| | Titanic | Sewol |
|--|---------|-------|
| Niños (0-12) | 57,5% (n=73) | 50,0% (n=4) |
| Adultos (13+) | 36,7% (n=818) | 33,5% (n=439) |
| Ratio niños/adultos | **1,57x** | **1,49x** |

En este caso los dos ratios son similares y próximos a 1,5x. En el Titanic, el protocolo "niños primero" se refleja en los datos. En el Sewol, el pequeño grupo de niños disponible en la muestra muestra un ratio similar. No obstante, **hay que interpretar el resultado del Sewol con extrema cautela**: con solo 4 niños en la muestra (2 supervivientes, 2 no supervivientes), el 50% es estadísticamente no significativo y no permite extraer ninguna conclusión sobre el protocolo de evacuación para este grupo.

---

## 6. Tabla Comparativa Final

La tabla siguiente concentra los indicadores principales de los dos desastres:

| Indicador | Titanic (1912) | Sewol (2014) |
|-----------|---------------|--------------|
| Tasa global de supervivencia | 38,4% | 33,6% |
| Supervivencia mujeres | 74,2% | 29,1% |
| Supervivencia hombres | 18,9% | 36,8% |
| Ratio mujeres/hombres | 3,93x | 0,79x |
| Supervivencia adolescentes (13-19) | 45,3% | 23,9% |
| Supervivencia adulto joven (20-39) | 33,7% | 52,2% |
| Supervivencia adulto (40-59) | 40,4% | 78,2% |
| Supervivencia mayor (60+) | 26,9% | 46,2% |
| Edad media supervivientes | 28,2 años | 31,4 años |
| Edad media no supervivientes | 29,8 años | 20,8 años |

La edad media de los no supervivientes en el Sewol (20,8 años) es especialmente reveladora: confirma que los jóvenes —principalmente los estudiantes de instituto— fueron el grupo más afectado por la tragedia.

---

## 7. Conclusiones

### 7.1 Respuesta a la pregunta de investigación

El análisis comparativo entre el Titanic y el Sewol permite extraer las siguientes conclusiones:

1. **El protocolo de evacuación determina los patrones de supervivencia por encima de las variables sociodemográficas.** En el Titanic, el género fue el factor más determinante (ratio 3,93x). En el Sewol, este factor prácticamente desaparece o se invierte (ratio 0,79x), lo que indica que la variable género no operó de la misma manera en ausencia de un protocolo explícito de prioridad.

2. **La edad tuvo impactos opuestos en los dos desastres.** En el Titanic, los grupos de menor edad sobrevivieron más (consistente con "niños primero"). En el Sewol, los adolescentes —la mayoría a bordo— tuvieron la tasa de supervivencia más baja (23,9%), mientras que los adultos mayores sobrevivieron mucho más (78,2%). Esto es consistente con la hipótesis de que los estudiantes siguieron las instrucciones de permanecer en los camarotes, mientras que los adultos actuaron de manera más autónoma.

3. **Las tasas globales similares (38,4% vs 33,6%) esconden patrones internos radicalmente divergentes.** Una comparativa superficial podría concluir que los dos desastres tuvieron resultados similares; el análisis por subgrupos revela que los mecanismos que determinaron la supervivencia fueron completamente diferentes.

4. **El perfil de máximo riesgo es diferente en cada desastre.** En el Titanic: hombre, tercera clase, adulto mayor. En el Sewol: adolescente, independientemente del género.

### 7.2 Limitaciones del análisis

- El dataset del Sewol contiene 476 registros, de los cuales 443 se han utilizado como pasajeros. Se trata de una muestra relativamente pequeña para algunos subgrupos (especialmente niños y adultos jóvenes).
- El grupo de niños en el Sewol (n=4) no permite ninguna inferencia estadística válida.
- No disponemos de datos sobre la ubicación exacta de cada pasajero en el barco en el momento del accidente, lo que limita el análisis de los factores físicos de la evacuación.
- Los datos del Sewol no incluyen información sobre la clase o el nivel socioeconómico de los pasajeros, a diferencia del Titanic, lo que impide controlar esta variable en la comparativa.
- Los dos desastres están separados por más de 100 años y se produjeron en contextos culturales y geográficos muy diferentes. Las conclusiones deben interpretarse con esta perspectiva.

---

## 8. Outputs Generados

### 8.1 Dataset combinado

Fichero: `output/comparativa_titanic_sewol.csv`

Contiene 1.334 registros (891 del Titanic train + 443 pasajeros del Sewol) con las columnas normalizadas: `disaster`, `gender`, `age`, `survived`, `role`.

### 8.2 Visualizaciones

| Fichero | Descripción |
|---------|-------------|
| `output/comparativa_genere_titanic_sewol.png` | Barras agrupadas: supervivencia por género y desastre |
| `output/comparativa_edat_titanic_sewol.png` | Líneas: supervivencia por franja de edad y desastre |
| `output/ratio_genere_titanic_sewol.png` | Barras: ratio mujeres/hombres por desastre |
| `output/ratio_nens_adults_titanic_sewol.png` | Barras: supervivencia niños vs adultos por desastre |
| `output/barres_comparativa_titanic_sewol.png` | Barras horizontales: tabla comparativa visual |

---

## 9. Tecnologías y Herramientas Utilizadas

| Herramienta | Versión | Uso en este bloque |
|-------------|---------|-------------------|
| **Python** | 3.14.0 | Lenguaje principal |
| **pandas** | 3.0.3 | Manipulación y normalización de datos |
| **NumPy** | 2.4.5 | Operaciones numéricas |
| **matplotlib** | — | Visualizaciones |
| **Jupyter Notebook** | — | Entorno de desarrollo y documentación |

---

*Documento generado como parte del TFE — Posgrado en Business Analytics, UPC School · Mayo 2026*
