# Update del equipo — Mayo 2026
**De: Albert · Para: Pablo y Roger**

---

Hola chicos, os dejo aquí todo lo que tengo hecho para que podáis cogerlo y seguir sin tener que preguntarme nada básico. Si tenéis dudas igual me escribís, pero debería estar todo bastante claro.

---

## Qué he hecho

### Los datos

Descargué los datos del Titanic directamente de Kaggle (los tenéis en `titanic_kaggle/`). Son tres archivos: el de entrenamiento con 891 pasajeros que sí tienen el dato de si sobrevivieron o no, el de test con 418 pasajeros sin esa etiqueta, y un ejemplo del formato de entrega.

Los datos del Sewol ya estaban en `sewol/sewol_eng.csv`. Tiene género, edad, supervivencia, piso del barco y posición. Bastante usable para la comparativa.

---

### El notebook

Todo el análisis está en `notebooks/TFE_Titanic_ML.ipynb`. Está ejecutado, así que cuando lo abráis ya veréis los resultados y los gráficos sin tener que correr nada.

Básicamente el notebook hace esto, por secciones:

**Sección 1 — Carga de datos.** Se cargan los archivos, se describe cada columna y se ve la pinta general de los datos.

**Sección 2 — EDA (análisis exploratorio).** Aquí está la parte visual. Se ve cómo varía la supervivencia según el género, la clase, la edad, el tamaño de la familia, el puerto de embarque y la tarifa. Hay bastantes gráficos que se guardan en `output/`.

**Sección 3 — Limpieza y variables nuevas.** Se rellenan los huecos que faltan (la edad con la mediana por grupos, el puerto con el más frecuente...) y se crean 8 variables nuevas a partir de las originales. La más interesante es el título social (Mr, Mrs, Miss, Master...) que se extrae del nombre de cada pasajero y resume de golpe género, edad y posición social.

**Sección 4 — El modelo.** Se prueban dos modelos: una regresión logística como base de comparación, y un Random Forest que es el que se usa al final. El Random Forest acierta el 82% de los casos. También están los valores SHAP, que son los que explican cuánto contribuye cada variable a cada predicción.

**Sección 5 — Outputs.** Se genera el archivo de submission para Kaggle y el CSV enriquecido para el equipo.

---

### Los resultados

Estos son los números más importantes que salen del análisis:

- Tasa de supervivencia global: **38,4%** (342 de 891 pasajeros)
- Mujeres: **74,2%** de supervivencia
- Hombres: **18,9%** de supervivencia
- 1ª clase: **63%** · 2ª clase: **47,3%** · 3ª clase: **24,2%**
- Mujer + 1ª clase: **96,8%** ← el perfil más extremo
- Hombre + 3ª clase: **13,5%** ← el otro extremo
- El modelo acierta el **82,1%** de los casos

El factor que más pesa en el modelo es el **género** (28,7% del peso total). Después viene el título social, la tarifa y la edad. La clase del billete tiene menos peso del que se podría esperar porque ya queda capturada en parte por la tarifa y el título.

---

### Los archivos que os importan

**`output/titanic_enriched.csv`** — Este es el archivo clave para vosotros. Tiene los 1.309 pasajeros (train + test) con todas las variables originales, las 8 variables nuevas que creé, y las probabilidades de supervivencia que calculó el modelo. Está listo para meterlo en Power BI o para cruzarlo con el Sewol directamente.

**`output/*.png`** — Todos los gráficos del análisis ya generados. Podéis pegarlos en el Word del informe sin tener que hacer nada.

**`output/titanic_submission.csv`** — El archivo para subir a Kaggle. Ya tiene el formato correcto. Solo hay que entrar a la competición y subirlo.

---

### El simulador y el dashboard

Hice dos páginas web en `simulator/`:

**`index.html`** — El simulador interactivo. Responde 5 preguntas (género, nivel económico, edad, compañía y puerto de embarque) y te dice tu probabilidad estimada de supervivencia. Hay un modo "datos" que enseña las variables reales del modelo. Se abre directamente en el navegador, no necesita nada más.

**`model_explorer.html`** — Una especie de dashboard académico del modelo. Tiene los KPIs principales, los gráficos del EDA, la tabla de importancia de variables, la comparación de modelos, un ranking de perfiles de supervivencia con el top 10 y el bottom 10, y la sección de limitaciones. Está pensado para enseñarlo al tribunal y que quede claro qué se ha hecho.

Los dos están conectados entre sí con botones.

---

### La documentación

En `docs/` hay cuatro archivos:

- `informe_bloque_ML_Albert.md` — Mi sección del informe. Son unas 10 páginas sobre el dataset, la metodología, los resultados y las conclusiones. Está listo para integrarlo en el Word del informe final.
- `GUIA_COMPANEROS.md` — Instrucciones más detalladas de cómo usar el CSV, código Python de ejemplo para la comparativa con el Sewol, y sugerencias de páginas para el dashboard de Power BI.
- `CHEATSHEET_DEFENSA.md` — Mi guía para la presentación. Tiene los números que hay que saber de memoria, posibles preguntas del tribunal con respuestas preparadas, y frases que quedan bien. Si queréis echarle un ojo por si os sirve de referencia para vuestra parte.
- Este archivo (el que estáis leyendo).

---

## Para Pablo — comparativa con el Sewol

El CSV que te dejo (`titanic_enriched.csv`) tiene estas columnas que te interesan: `Sex`, `Age`, `Survived`, `Pclass`, `SurvivalProb`, `AgeGroup`, `FamilyCat`.

El Sewol tiene `gender`, `age`, `Raw` (que es "survival" o "Dead"), `floor` y `location`. Eso te permite comparar directamente género y edad entre los dos desastres. Lo que va a ser interesante es ver si el patrón de género se mantiene o no, porque en el Sewol la mayoría de víctimas eran estudiantes de instituto y el protocolo de evacuación fue muy diferente (les dijeron que se quedaran quietos).

Una idea que creo que quedaría muy bien: una tabla simple con tres columnas — variable, tasa Titanic, tasa Sewol — para género y grupos de edad. Si el patrón se invierte o cambia mucho, eso ya es la conclusión central del trabajo.

En `GUIA_COMPANEROS.md` tienes código Python directo para hacer esa comparativa, copiar y pegar.

---

## Para Roger — el dashboard de Power BI

El archivo `titanic_enriched.csv` se importa directamente en Power BI sin necesidad de transformaciones. Tiene columnas listas para usar: `SurvivalProb` (probabilidad continua de 0 a 1), `ModelPred` (predicción binaria), `Split` (para distinguir train de test), y todas las variables categóricas ya limpias.

Los gráficos del `output/` se pueden incrustar directamente en el Word del informe como imágenes. Los más visuales son el heatmap de género × clase y el SHAP summary.

Para el informe, mi sección en `informe_bloque_ML_Albert.md` está lista para integrar. Cubre el dataset, la metodología, los resultados y las conclusiones. Solo haría falta ajustar el estilo si el grupo decide usar un formato concreto.

---

## Ideas para los próximos pasos

Cosas que me parecen interesantes para enriquecer el trabajo si tenéis tiempo:

**Comparativa de protocolos de evacuación.** En el Titanic fue "mujeres y niños primero"; en el Sewol les dijeron a los pasajeros que se quedaran en sus camarotes. Si eso aparece reflejado en los datos (por ejemplo, si en el Sewol el género tiene menos impacto que en el Titanic), es una conclusión muy potente para el trabajo.

**Un mapa de calor en Power BI de género × clase para el Titanic.** Como el que tengo en el explorer, pero interactivo. Quedaría muy bien y explicaría visualmente el punto principal del trabajo.

**Tabla comparativa final sencilla.** Algo del estilo: "En el Titanic, ser mujer multiplicaba por 3,9 la probabilidad de sobrevivir. En el Sewol, ese factor era X." Una sola tabla así en el informe dice más que tres páginas de análisis.

**Citar algún paper del Titanic.** Hay análisis académicos del naufragio (Hall 1986, Frey et al. 2011) que estudian exactamente estas variables. Si mencionáis en el informe que vuestros resultados son consistentes con la literatura, el tribunal lo valorará mucho.

---

## Fechas

- **11 junio** — Entrega de la documentación
- **16 y 18 junio** — Presentaciones

Nos quedan menos de cuatro semanas. Yo estoy disponible si necesitáis que ajuste algo del CSV o que añada alguna columna extra. También puedo repasar vuestra parte del informe si queréis un segundo ojo antes de entregar.

Cualquier duda, ya sabéis.

— Albert
