# Update del bloc de ML — Maig 2026
**Albert**

---

Hola, us deixo aquí tot el que he fet perquè ho pugueu pillar rapid. He intentat documentar-ho bé si teniu dubtes ja m'escriviu.

---

## Què he fet

### 1. Les dades

He descarregat el dataset del Titanic directament de Kaggle. El teniu a `titanic_kaggle/`:

- `train.csv` — 891 passatgers **amb** l'etiqueta de si van sobreviure o no. És el que s'ha usat per entrenar i avaluar el model.
- `test.csv` — 418 passatgers **sense** etiqueta. El model els ha de predir.
- `gender_submission.csv` — exemple del format que demana Kaggle per al lliurament. Tambe ens he presentat al concurs de kaggle aviam que diuen ells 

El dataset del Sewol ja estava a `sewol/sewol_eng.csv`. Té les columnes `gender`, `age`, `Raw` (que és "survival" o "Dead"), `floor` i `location`. S' haura d' adaptar i crear auxiliar en els dos datasets (titanic i Sewol) per poderho creuar guay i treure conclusions (evacuacio etc)

---

### 2. El notebook — `notebooks/TFE_Titanic_ML.ipynb`

Aquí és on hi ha tota la feina tècnica. Està **executat**, o sigui que quan l'obriu ja veureu tots els resultats i gràfics sense haver de córrer res. Té cinc seccions:

**Secció 1 — Càrrega i descripció del dataset.** S'explica cada columna, els tipus de dades i una primera visió dels 891 passatgers.

**Secció 2 — Anàlisi exploratòria (EDA).** Es mira com varia la supervivència en funció del gènere, la classe, l'edat, la mida familiar, el port d'embarcament i la tarifa. Tots els gràfics es guarden a `output/` i estan llestos per posar al Word.

**Secció 3 — Neteja i variables noves.** S'omplen els buits que manquen (l'edat, el port, la tarifa) i es creen 8 variables noves a partir de les originals. La més important és el títol social (Mr, Mrs, Miss, Master, Rare), que s'extreu del nom de cada passatger i combina en una sola variable el gènere, l'edat aproximada i la posició social.

**Secció 4 — El model.** Es proven dos models: Regressió Logística com a referència, i Random Forest com a model principal. El Random Forest encerta el 82% dels casos. S'inclou també l'anàlisi SHAP, que explica quant contribueix cada variable a cada predicció individual. Diem que em fet els dos models i hem ist que el random es millor tal

**Secció 5 — Outputs.** Es genera el fitxer per a Kaggle i el CSV enriquit per creuarho despres amb el Sewol.

---

### 3. Els resultats

Aquests són els números del Titanic:

| Dada | Valor |
|------|-------|
| Taxa global de supervivència | **38,4%** (342 de 891 passatgers) |
| Dones | **74,2%** de supervivència |
| Homes | **18,9%** de supervivència |
| 1a classe | **63,0%** |
| 2a classe | **47,3%** |
| 3a classe | **24,2%** |
| Dona + 1a classe | **96,8%** ← perfil més positiu |
| Home + 3a classe | **13,5%** ← perfil més negatiu |
| Precisió del model | **82,1%** |
| ROC-AUC | **85,6%** |

El factor més determinant és el **gènere** (28,7% del pes total del model), seguit del títol social, la tarifa i l'edat.

---

### 4. Els fitxers que us importen

**`output/titanic_enriched.csv`** — Aquest és el fitxer clau per a vosaltres. Té els 1.309 passatgers (train + test) amb totes les variables originals, les 8 variables noves que he creat, i les **probabilitats de supervivència** que ha calculat el model per a cada passatger. Està llest per importar-lo a Power BI o creuar-lo amb el Sewol directament.

**`output/*.png`** — Tots els gràfics de l'anàlisi ja generats, ja els fotrem a la memoria 


---

### 5. Les eines web — la part que crec que és més potent per a la presentació

Aquí és on crec que hi ha el valor afegit més gran del meu bloc, perquè és el que el tribunal veurà i el que la gent entén de seguida. He creat dues pàgines web que obren directament al navegador, sense necessitat d'instal·lar res.

---

#### `simulator/index.html` — El simulador interactiu

**"Hauries sobreviscut al Titanic?"**

És una pàgina amb una experiència completa. L'usuari respon 5 preguntes i rep la seva probabilitat estimada de supervivència amb una explicació personalitzada de per què li ha sortit aquell percentatge.

Les 5 preguntes estan directament lligades a les variables del model:

1. **Gènere** → variable `Sex_num`
2. **Quant cobres al mes?** → la resposta es tradueix a `Pclass` i `Fare` estimada
3. **Quants anys tens?** → variables `Age` i `AgeGroup`
4. **Amb qui viatges?** → `FamilySize`, `IsAlone`, `FamilyCat`
5. **Des d'on embarcaries?** → `Embarked`


El resultat mostra:
- Un **indicador circular animat** amb el percentatge
- Un **veredicte** (les probabilitats estaven a favor teu / en contra teva)
- Una **explicació narrativa** de per què li ha sortit aquell número, citant el pes real de cada variable
- Un **desglossament per factors** (gènere, classe, edat, família, port) amb el pes real de cada un extret del model
- Un **Mode dades** que ensenya les variables reals tal com les veuria el model: `Sex = female`, `Pclass = 1`, `Age = 27`, etc. Això és el que li va bé al tribunal per veure que el simulador no és màgia sinó el model real.
- Una **cita històrica** del naufragi

Important a destacar és que **tots els càlculs estan basats en les taxes historials reals** del dataset, no en números inventats.

---

#### `simulator/model_explorer.html` — El dashboard acadèmic

Aquesta és la peça que explica el model de forma acadèmica. Pensada per obrir davant el tribunal i guiar l'explicació. Té les seccions següents:

**KPIs principals** — Una graella amb els 9 números més importants del projecte: passatgers analitzats, variables del model, taxa global, supervivència per gènere, precisió, ROC-AUC. Hi ha també una taula de calor Gènere × Classe amb les 6 taxes historials reals (96,8%, 92,1%, 50%, 36,9%, 15,7%, 13,5%).

**Visualitzacions EDA** — 8 gràfics del notebook amb descriptions. Es pot clicar qualsevol per ampliar-lo a pantalla completa.

**Taula de variables** — Les 14 variables del model amb la seva descripció, si és original o derivada, la direcció de l'efecte (positiva o negativa per a la supervivència) i el pes en el model.

**Importància de factors** — Barres animades amb els valors reals del Random Forest. El gènere és el primer (28,7%), seguit del títol social (14,7%), la tarifa (12,7%), etc.

**Rendiment del model** — Comparativa entre els dos models, amb les matrius de confusió i les corbes ROC com a imatges clicables.

**Ranking de perfils de supervivència** — Aquí és on crec que el tribunal agraïrà molt tenir-ho visualitzat. Hi ha dues peces:

  - Una **taula de ranking** amb 18 perfils representatius ordenats de major a menor probabilitat, amb barres de color. Va del 97% (dona · 1a classe · adulta jove · família petita) fins al 3% (home · 3a classe · gran · família nombrosa).
  - Un **Top 10 / Bottom 10** en dues columnes: els 10 perfils amb major probabilitat i els 10 amb menor. Cada un té el percentatge i un insight al peu ("Els 8 primers perfils són exclusivament dones" / "Ni tan sols ser de 1a classe compensava ser home gran amb família nombrosa").

**Limitacions del model** — 8 targetes amb les limitacions honestes del treball: el dataset és petit, hi ha valors que manquen, les variables no mesuren exactament el que expliquen, el model identifica correlacions però no causalitat, etc.

Les dues pàgines estan connectades amb botons: des del simulador hi ha un botó "Veure l'anàlisi complet del model" que obre l'explorer, i a l'inrevés.

---

### 6. La documentació

A `docs/` hi ha quatre fitxers:

- `informe_bloque_ML_Albert.md` — La meva secció de la memòria del TFE. Unes 10 pàgines que cobreixen el dataset, la metodologia, els resultats i les conclusions. Llesta per integrar al Word.
- `GUIA_COMPANEROS.md` — Instruccions detallades de com usar el CSV, codi Python d'exemple per a la comparativa amb el Sewol, i suggeriments de pàgines per al dashboard de Power BI amb mesures DAX.
- `CHEATSHEET_DEFENSA.md` — La meva guia per a la presentació. Té els números que cal saber de memòria, preguntes probables del tribunal amb respostes preparades, i frases que queden bé.
- `UPDATE_EQUIPO.md` — El document que esteu llegint ara.

---

## Proxims pasos persona 2 — comparativa amb el Sewol

El CSV que et deixo (`titanic_enriched.csv`) té les columnes `Sex`, `Age`, `Survived`, `Pclass`, `SurvivalProb`, `AgeGroup` i `FamilyCat`, que pots creuar directament amb les del Sewol.

El que crec que serà més interessant és comparar l'efecte del gènere entre els dos desastres. Al Titanic les dones van sobreviure al 74,2%; al Sewol, la majoria de víctimes eren estudiants d'institut i el protocol va ser molt diferent (els van dir que es quedessin quiets als seus camarots). Si el patró de gènere canvia o s'inverteix, ja tens la conclusió central del treball.
S'hauran de crear columnes noves etc per creuarho millor.

A `GUIA_COMPANEROS.md` tens codi Python directe per fer aquesta comparativa.

---

## Proxims pasos persona 3 — el dashboard de Power BI o fer un historic de visuals o com vulgueu

El fitxer `titanic_enriched.csv` s'importa directament a Power BI sense transformacions. Les columnes interessants per al dashboard:

- `SurvivalProb` — probabilitat contínua de 0 a 1 (útil per a gràfics de dispersió)
- `ModelPred` — predicció binària del model (0 o 1)
- `Split` — indica si el passatger és del conjunt train o test
- `AgeGroup`, `FamilyCat`, `FareBand` — categories netes per a filtres i visuals

Els gràfics de `output/` es poden enganxar directament al Word. El més potent visualment és el heatmap de gènere × classe (`eda_heatmap_genero_clase.png`) i el SHAP summary (`shap_summary.png`).

---

## Idees per als pròxims passos

Coses que crec que molarien:

**La comparativa de protocols d'evacuació** és la conclusió més potent que pot tenir el treball. Si els dades del Sewol mostren que el gènere va tenir menys impacte que al Titanic, és perquè el context va ser totalment diferent. Això li dona molt de valor al treball respecte a un simple anàlisi del Titanic.

**Una taula comparativa final sencilla** a la memòria, tipus: "Al Titanic, ser dona multiplicava per 3,9 la probabilitat de sobreviure. Al Sewol, aquest factor va ser X." Una taula d'aquestes, ben posada a les conclusions, val més que tres pàgines d'anàlisi. Podriem entrar o estudiar els insights fins i tot filosoficament `"Ha canviat la generacio la "educacio" de mujeres primero?"`

**Citar un parell de papers del Titanic** a la memòria (Hall 1986, Frey et al. 2011) que estudien exactament aquestes variables. Si els vostres resultats coincideixen amb la literatura, ens va perfecte.

---

## Dates

- **11 de juny** — Lliurament de la documentació
- **16 i 18 de juny** — Presentacions


— Albert

---

# Update del bloque de Comparativa — Mayo 2026
**Pablo**

---

Hola, os dejo el resumen de lo que he hecho en el bloque de comparativa Titanic vs Sewol. Todo está ejecutado y listo.

---

## Qué he hecho

### 1. La normalización de los datasets

He creado un dataset unificado a partir del `titanic_enriched.csv` de Albert y del `sewol_eng.csv`. He normalizado las columnas equivalentes (género, edad, supervivencia) y he creado una columna `role` para separar tripulación y pasajeros. He utilizado solo los 891 pasajeros del train del Titanic (los que tienen supervivencia real) y los 443 pasajeros del Sewol (excluyendo los 33 tripulantes).

El fichero combinado está en `output/comparativa_titanic_sewol.csv`.

### 2. Los notebooks

Todo el trabajo técnico está en dos notebooks:

**`notebooks/TFE_Comparativa_Sewol.ipynb`** — El trabajo de análisis principal. Tiene cinco secciones:

- **Paso 1** — Carga y descripción de los dos datasets con todas las columnas.
- **Paso 2** — Normalización y unificación en un DataFrame combinado. Incluye la creación de la columna `role` y la imputación de nulos de edad del Sewol con la mediana.
- **Paso 3** — Análisis de supervivencia por género. Tabla y gráfico comparativo. Resultado principal: ratio mujeres/hombres de 3,93x en el Titanic vs 0,79x en el Sewol (el patrón se invierte).
- **Paso 4** — Análisis de supervivencia por franja de edad. Grupos adaptados a la distribución del Sewol (69% adolescentes). Los adolescentes del Sewol tuvieron la tasa más baja: 23,9%.
- **Paso 5** — Exportación del dataset combinado.

**`notebooks/TFE_Conclusions_Comparativa.ipynb`** — Las conclusiones visuales. Tiene tres secciones:

- **Paso 1** — Carga del CSV combinado.
- **Paso 2** — Tabla comparativa final con todos los indicadores clave, tabla con colores y gráfico de barras horizontales.
- **Paso 3** — Análisis del protocolo de evacuación: ratio mujeres/hombres y niños/adultos para los dos desastres.

### 3. Los resultados principales

| Dato | Titanic | Sewol |
|------|---------|-------|
| Tasa global | **38,4%** | **33,6%** |
| Mujeres | **74,2%** | **29,1%** |
| Hombres | **18,9%** | **36,8%** |
| Ratio mujeres/hombres | **3,93x** | **0,79x** ← el patrón se invierte |
| Adolescentes (13-19) | **45,3%** | **23,9%** ← el grupo más perjudicado |
| Adultos (40-59) | **40,4%** | **78,2%** |
| Edad media no supervivientes | **29,8 años** | **20,8 años** |

La conclusión central: las mismas variables —género y edad— operaron de manera completamente diferente en cada desastre. El contexto y el protocolo de evacuación importa más que las normas sociales.

### 4. Los ficheros que os interesan

**`output/comparativa_titanic_sewol.csv`** — El dataset combinado normalizado. 1.334 filas, columnas: `disaster`, `gender`, `age`, `survived`, `role`.

**`output/*.png`** — Cinco gráficos comparativos listos para la memoria:
- `comparativa_genere_titanic_sewol.png`
- `comparativa_edat_titanic_sewol.png`
- `ratio_genere_titanic_sewol.png`
- `ratio_nens_adults_titanic_sewol.png`
- `barres_comparativa_titanic_sewol.png`

### 5. La documentación

En `docs/` hay dos ficheros nuevos:

- `informe_bloque_comparativa_Pablo.md` — Mi sección de la memoria del TFE. Cubre la metodología, los resultados y las conclusiones. Lista para integrar en el Word.
- `CHEATSHEET_Pablo.md` — Mi guía para la presentación. Tiene los números clave, preguntas probables del tribunal con respuestas preparadas, y frases para la defensa.

---

— Pablo
