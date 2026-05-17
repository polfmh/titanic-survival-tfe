# Cheatsheet de Defensa — Comparativa Titanic vs Sewol
### TFE · PBA · UPC School · Pablo · 2026

---

## GUIÓN DE PRESENTACIÓN (3 minutos para tu parte)

**Frase de apertura:**
> "El bloque anterior nos ha dicho qué factores determinaron la supervivencia en el Titanic. Mi trabajo era responder: ¿esos mismos factores explican lo que pasó en el Sewol en 2014? La respuesta es no, y os lo demuestro con los datos."

**Flujo recomendado:**
1. Enseña la tabla comparativa final → párate en el ratio mujeres/hombres
2. Di: *"En el Titanic, ser mujer multiplicaba por 3,93 la probabilidad de sobrevivir. En el Sewol, el ratio es 0,79: los hombres sobrevivieron más que las mujeres."*
3. Enseña el gráfico de barras de género → contrasta visualmente los dos desastres
4. Enseña el gráfico de franjas de edad → destaca los adolescentes del Sewol (23,9%)
5. Di: *"La edad media de los que murieron en el Sewol es de 20,8 años. El grupo más afectado fueron los estudiantes de instituto."*
6. Cierra con la conclusión: *"Las mismas variables —género y edad— operaron de manera completamente diferente en cada desastre. Eso nos dice que el protocolo de evacuación importa más que las normas sociales."*

---

## NÚMEROS QUE TIENES QUE SABER DE MEMORIA

| Dato | Valor | Para qué sirve |
|------|-------|----------------|
| Tasa global Titanic | **38,4%** | Punto de partida |
| Tasa global Sewol | **33,6%** | Similar al Titanic, pero por razones muy diferentes |
| Mujeres Titanic | **74,2%** | Protocolo "mujeres primero" aplicado |
| Mujeres Sewol | **29,1%** | Protocolo de género NO aplicado |
| Hombres Titanic | **18,9%** | Contraste con las mujeres |
| Hombres Sewol | **36,8%** | Los hombres sobrevivieron MÁS que las mujeres |
| Ratio mujeres/hombres Titanic | **3,93x** | El factor clave de contraste |
| Ratio mujeres/hombres Sewol | **0,79x** | El patrón se invierte |
| Adolescentes Titanic (13-19) | **45,3%** | Referencia |
| Adolescentes Sewol (13-19) | **23,9%** | El grupo más perjudicado del Sewol |
| Adultos (40-59) Sewol | **78,2%** | Los adultos sobrevivieron mucho más |
| Edad media no supervivientes Sewol | **20,8 años** | El dato más impactante del bloque |
| Pasajeros analizados | **891** Titanic + **443** Sewol | |

---

## PREGUNTAS PROBABLES DEL TRIBUNAL — CON RESPUESTAS

### Sobre la metodología comparativa

**¿Cómo habéis hecho comparable el Titanic con el Sewol si los datasets son muy diferentes?**
> "Hemos creado un DataFrame unificado normalizando las columnas equivalentes: género, edad y supervivencia existían en los dos datasets. La diferencia era el formato —en el Sewol la supervivencia estaba como texto 'survival'/'Dead'— que hemos traducido a 0 y 1. También hemos creado una columna 'role' para separar tripulación y pasajeros, y hemos trabajado siempre con pasajeros para hacer una comparación justa."

**¿Por qué habéis excluido la tripulación del Sewol?**
> "El dataset del Titanic no incluye tripulación —es exclusivamente de pasajeros. Si hubiéramos incluido los 33 tripulantes del Sewol, estaríamos comparando perfiles incomparables. Hemos preferido una comparativa limpia entre pasajeros de los dos barcos."

**¿Por qué habéis usado solo el conjunto 'train' del Titanic?**
> "El conjunto de test del Titanic no tiene el valor real de 'Survived' —es el que Kaggle reserva para la competición. Como necesitábamos supervivencias reales para calcular tasas, hemos usado exclusivamente los 891 pasajeros del train."

**¿Cómo habéis tratado los grupos de edad si la distribución es tan diferente?**
> "En el Sewol, el 69% de los pasajeros tenía entre 10 y 20 años —eran estudiantes de instituto. Con los grupos estándar de Albert, el grupo 'adolescente' habría concentrado casi todos los datos del Sewol y los demás grupos habrían quedado vacíos. Hemos adaptado los grupos a la distribución real del Sewol sin modificar ningún dato original."

### Sobre los resultados

**¿Cómo se explica que el ratio mujeres/hombres se invierta en el Sewol?**
> "No podemos afirmarlo con certeza absoluta porque no tenemos datos directos sobre las instrucciones dadas. Pero los datos apuntan claramente a que el protocolo de género 'mujeres primero' no se aplicó en el Sewol como en el Titanic. La hipótesis de nuestro trabajo es que las instrucciones de permanecer en los camarotes afectaron de manera indiscriminada, independientemente del género."

**¿Por qué los adultos mayores del Sewol sobrevivieron tanto (78,2%)?**
> "Probablemente porque eran una minoría pequeña —no estudiantes— con mayor autonomía y capacidad de decisión propia. A diferencia de los adolescentes, que eran 327 personas en un contexto de autoridad escolar, los adultos tenían menos presión para seguir instrucciones colectivas."

**Las tasas globales del Titanic y el Sewol son similares (38,4% vs 33,6%). ¿Significa que los dos desastres fueron iguales?**
> "Precisamente no. Esta similitud superficial esconde patrones internos radicalmente opuestos. Las tasas globales similares se producen por razones completamente diferentes: en el Titanic, las mujeres elevaron mucho la tasa global; en el Sewol, los adultos la elevaron pero los adolescentes (la mayoría) la bajaron. El resultado final es similar pero el mecanismo es el opuesto."

**¿Por qué la edad media de los no supervivientes del Sewol es tan baja (20,8 años)?**
> "Porque la mayoría de las víctimas eran los estudiantes de instituto —con edades entre 16 y 18 años. Este dato confirma numéricamente lo que los gráficos muestran visualmente: los jóvenes fueron el grupo más afectado por la tragedia."

### Sobre las limitaciones

**El grupo de niños en el Sewol tiene un 50% de supervivencia. ¿Significa que el protocolo 'niños primero' se aplicó?**
> "No se puede concluir nada de este resultado. El Sewol tiene solo 4 niños en la muestra: 2 sobrevivieron y 2 no. Con un tamaño de muestra tan pequeño, el 50% no tiene ningún significado estadístico. Hemos incluido el resultado por completitud pero lo hemos marcado explícitamente como no significativo."

**¿Podríais haber incluido la clase social como variable comparativa?**
> "En el Titanic sí teníamos la clase (1ª, 2ª, 3ª). En el Sewol no hay ninguna variable equivalente —todos eran pasajeros de la misma excursión. Sin equivalente en el Sewol, no podemos hacer la comparativa y hemos preferido no forzar una variable que no existe."

**¿El dataset del Sewol es suficientemente grande para conclusiones estadísticamente sólidas?**
> "Para el análisis de género, sí: 182 mujeres y 261 hombres es un tamaño aceptable. Para las franjas de edad, algunos grupos del Sewol son pequeños —especialmente niños y adultos jóvenes— y sus tasas deben interpretarse con cautela. Lo documentamos explícitamente en las limitaciones del informe."

---

## FRASES POTENTES PARA LA PRESENTACIÓN

- *"Las mismas variables —género y edad— operaron de manera completamente diferente en cada desastre. El contexto lo cambia todo."*
- *"En el Titanic, ser mujer multiplicaba por 3,93 la probabilidad de sobrevivir. En el Sewol, los hombres sobrevivieron más. En cien años, el factor determinante había cambiado completamente."*
- *"La edad media de los que murieron en el Sewol era de 20,8 años. No es una estadística: es el perfil de una clase de instituto."*
- *"Las tasas globales del 38% y el 34% parecen similares. Pero llegar al mismo resultado por razones completamente opuestas es la conclusión más importante de este trabajo."*
- *"El modelo de Albert predecía un 82% de aciertos en el Titanic. Las mismas variables aplicadas al Sewol habrían fallado para los adolescentes —no porque el modelo fuera malo, sino porque el contexto era diferente."*

---

## LO QUE NO TIENES QUE MENCIONAR (a menos que pregunten)

- El nombre exacto de los ficheros CSV
- Los detalles técnicos de pandas o matplotlib
- La diferencia entre `pd.cut` y `pd.qcut`
- El número exacto de registros con nulos en el Sewol
- Los detalles de la normalización de columnas

---

*Pablo · TFE PBA · UPC School · 2026*
