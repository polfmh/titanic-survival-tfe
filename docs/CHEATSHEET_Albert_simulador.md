# Cheatsheet de Defensa — Simulador Titanic
### TFE · PBA · UPC School · Albert Felis · 2026

---

## GUIÓN DE PRESENTACIÓN (3 minutos para tu parte)

**Frase de apertura:**
> "Para responder qué factores determinan la supervivencia en un desastre marítimo, entrené un modelo con los datos reales de los 891 pasajeros del Titanic. Os voy a mostrar los resultados directamente en el simulador."

**Flujo recomendado:**
1. Abre `index.html` → muestra la landing (menciona el proyecto y el equipo)
2. Haz el quiz con un perfil dramático: **hombre, sueldo bajo, adulto, solo** → resultado bajo (~8%)
3. Vuelve a intentarlo con: **mujer, sueldo alto, joven, familia pequeña** → resultado alto (97%)
4. Di: *"La diferencia entre estos dos perfiles es de 89 puntos porcentuales. Eso no es azar, son los datos históricos del Titanic."*
5. Pulsa "Ver análisis completo" → abre `model_explorer.html`
6. Enseña el Ranking de perfiles → señala la tabla
7. Enseña el Top 10 / Bottom 10 → resalta el insight de las mujeres

---

## NÚMEROS QUE TIENES QUE SABER DE MEMORIA

| Dato | Valor | Para qué sirve |
|------|-------|----------------|
| Tasa global supervivencia | **38,4%** | Punto de partida del análisis |
| Supervivencia mujeres | **74,2%** | Factor género principal |
| Supervivencia hombres | **18,9%** | Contraste con mujeres |
| 1ª clase | **63,0%** | Impacto de la clase social |
| 3ª clase | **24,2%** | Impacto de la clase social |
| Mujer + 1ª clase | **96,8%** | Perfil extremo positivo |
| Hombre + 3ª clase | **13,5%** | Perfil extremo negativo |
| Pasajeros en el modelo | **891** train + 418 test |
| Precisión del modelo | **82,1%** | El modelo acierta 8 de cada 10 |
| Número de variables | **14** | (6 originales + 8 derivadas) |
| Variable más importante | **Género** (28,7% del peso total) |

---

## PREGUNTAS PROBABLES DEL TRIBUNAL — CON RESPUESTAS

### Sobre el modelo

**¿Qué modelo usasteis?**
> "Usamos un Random Forest: un conjunto de 200 árboles de decisión que votan sobre la predicción final. También comparamos con Regresión Logística como referencia. Elegimos el Random Forest porque generaliza mejor —lo comprobamos con validación cruzada— y porque nos permite explicar cada predicción individualmente con SHAP."

**¿Qué es la validación cruzada?**
> "Dividimos los datos en 5 bloques iguales. El modelo se entrena con 4 y se evalúa con el que queda fuera; esto se repite 5 veces. Así obtenemos una estimación más fiable de cómo funcionará el modelo con datos nuevos, en lugar de evaluarlo solo una vez."

**¿Qué precisión tiene el modelo?**
> "Un 82,1% de precisión en el conjunto de validación. Como referencia, predecir siempre 'no sobrevive' daría solo un 61,6% — el modelo mejora notablemente esa base. El ROC-AUC de 85,6% indica que distingue bien entre los dos grupos."

**¿Qué es el ROC-AUC?**
> "Es una métrica entre 0 y 1 que mide la capacidad del modelo para separar los dos grupos —los que sobrevivieron y los que no. Un valor de 0,5 sería puro azar; 1,0 sería perfecto. Nuestro 0,856 indica que el modelo discrimina muy bien."

**¿Por qué no usasteis un modelo más complejo?**
> "Con 891 muestras, un modelo demasiado complejo sobreajustaría —aprendería los datos de entrenamiento pero fallaría con datos nuevos. El Random Forest con los parámetros elegidos tiene el equilibrio adecuado. El objetivo no era maximizar el ranking de Kaggle sino tener un modelo interpretable que explique qué factores importan."

### Sobre los datos

**¿Qué hicisteis con los datos que faltaban?**
> "La columna de cabina tenía un 77% de valores nulos —demasiados para imputar— así que la convertimos en una variable binaria: '¿tiene cabina registrada sí o no?'. La edad faltaba en un 20% de los casos; en lugar de poner la media global, usamos la mediana por grupos: un niño varón (Master) no tiene la misma edad media que un señor adulto (Mr). El puerto de embarque solo tenía 2 nulos, los completamos con el más frecuente."

**¿Qué son las variables derivadas?**
> "Son variables que creamos a partir de las originales para que el modelo tenga información más útil. Por ejemplo: de los apellidos extraemos el título social (Mr, Mrs, Miss, Master) que combina género, edad y posición social en una sola variable. O combinamos SibSp y Parch para crear FamilySize —el tamaño total del grupo familiar."

**¿Por qué el puerto de embarque afecta a la supervivencia?**
> "Directamente no debería, pero en los datos hay una correlación: en Cherburgo embarcaron más pasajeros de primera clase. Es decir, el puerto actúa como indicador indirecto del nivel económico del pasajero. El modelo detecta esa correlación aunque no sea una causa directa."

### Sobre los resultados

**¿Cuál fue el factor más importante?**
> "El género, con un peso del 28,7% en el modelo. Las mujeres sobrevivieron al 74,2% y los hombres al 18,9%. El protocolo de evacuación 'mujeres y niños primero' es claramente visible en los datos y es el factor que más diferencia genera."

**¿Qué son los SHAP values?**
> "Es una técnica que nos permite ver, para cada pasajero concreto, cuánto contribuyó cada variable a su predicción. Por ejemplo, ser mujer empujó la probabilidad hacia arriba en X puntos; estar en tercera clase la bajó en Y puntos. Nos permite explicar el modelo caso por caso, en lugar de dar solo un número global."

**¿El modelo predice que todos los de 1ª clase sobrevivieron?**
> "No. El modelo predice probabilidades, no certezas. Un hombre de 1ª clase mayor, que viajaba solo, tenía solo un 13% de probabilidad según el modelo —y los datos históricos lo confirman: la clase sola no era suficiente si el género y la edad iban en contra."

**¿Cuánto vale un 82% de precisión?**
> "En contexto: si predijéramos que nadie sobrevive, acertaríamos el 61,6% de las veces —esa es la línea base. Nuestro modelo mejora eso 20 puntos. Scores por encima del 85% en esta competición generalmente implican haber visto los datos de test, lo que no tiene interés científico."

### Sobre el simulador

**¿Cómo funciona el simulador?**
> "Toma tus respuestas —género, nivel económico aproximado, edad, compañía y puerto de embarque— las traduce a las variables del modelo, y aplica las tasas históricas reales del Titanic ajustadas por el peso de cada variable. El resultado es una probabilidad estimada basada en los datos, no una predicción aleatoria."

**¿Las probabilidades del simulador coinciden exactamente con el modelo?**
> "Son una aproximación muy cercana. El simulador usa las tasas históricas reales por grupo (por ejemplo, mujer en 1ª clase → 96,8% base) y las ajusta por el efecto de edad y familia usando los pesos del modelo. La lógica es la misma, simplificada para funcionar sin Python en el navegador."

**¿Por qué el 'Modo datos' muestra Embarked=S siempre?**
> "Southampton era el puerto de salida principal y la elección más común. Para mantenerlo simple en la pantalla, si no se elige explícitamente, se asume Southampton como valor por defecto —es la moda del dataset."

### Sobre las limitaciones

**¿Qué limitaciones tiene el modelo?**
> "La principal es que el dataset es pequeño —891 muestras— lo que limita la complejidad del modelo. La columna de cabina tiene un 77% de nulos, lo que nos impide saber la ubicación exacta de cada pasajero en el barco. Además, el modelo identifica correlaciones, no causas: ser de primera clase correlaciona con sobrevivir porque los camarotes de primera clase estaban en cubiertas superiores más cercanas a los botes, pero el modelo no 'sabe' eso directamente."

**¿Por qué 3% es el mínimo y 97% el máximo?**
> "Es un límite que pusimos deliberadamente. Las probabilidades de 0% o 100% no existen en sistemas reales —siempre hay incertidumbre. El mínimo de 3% refleja que incluso en el peor perfil histórico, hubo supervivientes; el 97% que incluso en el mejor perfil, hubo víctimas."

---

## SI TE PREGUNTAN POR EL EXPLORER

**¿Qué es el ROC-AUC en la curva?**
> "La curva ROC muestra cómo varía el equilibrio entre detectar supervivientes reales (eje vertical) y equivocarse con los que no sobrevivieron (eje horizontal) al cambiar el umbral de decisión. El área bajo esa curva (AUC) resume ese equilibrio en un solo número: 0,856."

**¿Qué significa la tabla de importancia de variables?**
> "Muestra el peso que tuvo cada variable en las decisiones del modelo. Si el modelo tomó 1000 decisiones, el género intervino en el 28,7% de ellas de forma determinante. Las variables con menor peso —como el puerto de embarque (2,2%)— influyeron pero de forma más marginal."

**¿Por qué el título social tiene tanta importancia (14,7%)?**
> "Porque condensa mucha información: 'Master' identifica niños varones, que tenían prioridad; 'Mrs' y 'Miss' identifican mujeres; y los títulos raros como 'Dr' o 'Countess' señalan posición social elevada. Es una variable muy eficiente porque combina género, edad y clase en una sola."

---

## LO QUE NO TIENES QUE MENCIONAR (a menos que pregunten)

- La diferencia exacta entre Random Forest base y optimizado (son casi iguales)
- El número exacto de árboles (200) o profundidad máxima (6)
- La diferencia entre Gini y entropía como criterio de split
- El nombre de la librería exacta (scikit-learn) a menos que pregunten
- El proceso de búsqueda automática de parámetros en detalle

---

## FRASES POTENTES PARA LA PRESENTACIÓN

- *"El modelo no predice quién murió: nos dice qué perfiles tenían las probabilidades más bajas de llegar a un bote."*
- *"Ser hombre en tercera clase ya te dejaba en el 13%. Ser mayor y viajar con familia grande te llevaba al 3%."*
- *"Una mujer de primera clase tenía 7 veces más probabilidad de sobrevivir que un hombre de tercera."*
- *"El modelo acierta en 8 de cada 10 casos. Eso, con datos de hace 110 años, es un resultado sólido."*
- *"Las limitaciones forman parte del análisis: un modelo honesto sabe lo que no puede explicar."*

---

*Albert Felis · TFE PBA · UPC School · 2026*
