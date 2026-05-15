# Guía para el Equipo — Qué he hecho y cómo usar mis resultados
### TFE Supervivencia Marítima · Albert (Bloque ML Titanic) · Mayo 2026

---

## Lo que está listo y dónde encontrarlo

```
tfg_titanic/
├── titanic_kaggle/
│   ├── train.csv               ← Dataset original de Kaggle (891 pasajeros, con etiqueta)
│   ├── test.csv                ← Dataset original de Kaggle (418 pasajeros, sin etiqueta)
│   └── gender_submission.csv   ← Ejemplo de formato Kaggle
├── sewol/
│   └── sewol_eng.csv           ← Dataset Sewol (ya disponible)
├── notebooks/
│   └── TFE_Titanic_ML.ipynb   ← NOTEBOOK PRINCIPAL — todo el análisis aquí
├── output/
│   ├── titanic_enriched.csv    ← EL ARCHIVO CLAVE para el equipo
│   ├── titanic_submission.csv  ← Submission de Kaggle (ya hecho)
│   ├── eda_genero.png          ← Gráficos listos para el informe
│   ├── eda_clase.png
│   ├── eda_edad.png
│   ├── eda_familia.png
│   ├── eda_heatmap_genero_clase.png   ← Muy visual para el informe
│   ├── shap_summary.png        ← Importancia de variables (SHAP)
│   ├── shap_bar.png
│   ├── confusion_matrices.png
│   └── roc_curves.png
└── simulator/
    └── app.py                  ← Simulador interactivo (streamlit run app.py)
```

---

## El archivo clave: `output/titanic_enriched.csv`

Este es el archivo que el equipo necesita. Contiene **1.309 filas** (todos los pasajeros del dataset Titanic, train + test) y **22 columnas**.

### Columnas importantes

| Columna | Qué es | Para qué sirve |
|---------|--------|----------------|
| `PassengerId` | ID del pasajero | Identificación |
| `Survived` | 0/1 (real en train, predicción en test) | Variable objetivo |
| `SurvivalProb` | Probabilidad 0.0–1.0 de sobrevivir | **Para visualizaciones continuas** |
| `DeathProb` | 1 – SurvivalProb | |
| `ModelPred` | Predicción binaria del modelo | |
| `Split` | "train" o "test" | Para filtrar en Power BI |
| `Pclass` | 1, 2 o 3 | Clase del billete |
| `Sex` | male / female | Género |
| `Age` | Edad (imputada si era nula) | |
| `Title` | Mr / Mrs / Miss / Master / Rare | Título social |
| `FamilySize` | Nº de familiares + 1 | |
| `IsAlone` | 1 = viajaba solo | |
| `FamilyCat` | Solo / Familia pequeña / Familia grande | Para visualizaciones |
| `AgeGroup` | Niño / Adolescente / Adulto joven / Adulto / Mayor | Para visualizaciones |
| `FareBand` | Bajo / Medio-bajo / Medio-alto / Alto | Nivel económico |
| `HasCabin` | 1 = tiene cabina registrada | |
| `Embarked` | C / Q / S | Puerto de embarque |

---

## Para la Persona 2 — Análisis comparativo con el Sewol

### Qué tienes disponible del Titanic

Del `titanic_enriched.csv` puedes sacar directamente:
- Tasa de supervivencia por género
- Tasa de supervivencia por grupo de edad
- Distribución de supervivencia por tamaño familiar
- Probabilidades individuales de supervivencia

### Variables comunes con el Sewol

El dataset del Sewol (`sewol/sewol_eng.csv`) tiene estas columnas:
- `gender` (male/female) → equivalente a `Sex` en Titanic
- `age` → equivalente a `Age`
- `Raw` ("survival" / "Dead") → equivalente a `Survived`
- `floor` → piso del barco (sin equivalente directo en Titanic)
- `location` → posición en el barco (front/middle/back)
- `Category-1`, `Category-2`, `Category-3` → tipo de pasajero/tripulación

### Cómo hacer la comparativa

```python
import pandas as pd

# Cargar Titanic
titanic = pd.read_csv('output/titanic_enriched.csv')
titanic_train = titanic[titanic['Split'] == 'train'].copy()

# Cargar Sewol
sewol = pd.read_csv('sewol/sewol_eng.csv')
sewol['Survived_bin'] = (sewol['Raw'] == 'survival').astype(int)
sewol['Sex'] = sewol['gender']

# Tasas de supervivencia por género — Titanic
titanic_genero = titanic_train.groupby('Sex')['Survived'].mean()

# Tasas de supervivencia por género — Sewol
sewol_genero = sewol.groupby('Sex')['Survived_bin'].mean()

# Comparativa
print("SUPERVIVENCIA POR GÉNERO:")
print(f"  Titanic — Mujeres: {titanic_genero['female']*100:.1f}%  Hombres: {titanic_genero['male']*100:.1f}%")
print(f"  Sewol   — Mujeres: {sewol_genero.get('female', 0)*100:.1f}%  Hombres: {sewol_genero.get('male', 0)*100:.1f}%")
```

### Preguntas interesantes para la comparativa

1. **¿El patrón de género se mantiene?** En el Titanic las mujeres sobrevivieron al 74,2%. En el Sewol (alumnos de instituto), ¿hay diferencia por género?
2. **¿La posición en el barco influye?** El Sewol tiene variable `floor` y `location`. ¿Los pasajeros de pisos superiores sobrevivieron más?
3. **¿El contexto de evacuación importa?** El Titanic tuvo protocolo "mujeres primero"; el Sewol tuvo instrucciones contraproducentes de "quedarse en cabina". ¿Eso invierte los patrones?
4. **¿La edad sigue siendo relevante?** La mayoría de víctimas del Sewol eran adolescentes. ¿Cambia el perfil de edad respecto al Titanic?

### Ideas para visualizaciones comparativas

- **Gráfico de barras doble:** Tasa de supervivencia por género, Titanic vs. Sewol lado a lado.
- **Heatmap comparativo:** Género × Categoría de edad para ambos desastres.
- **Tabla resumen:** Tasa global, tasa mujeres, tasa hombres, factor de diferencia.
- **Conclusión narrativa:** "En el Titanic, ser mujer multiplicó por X la probabilidad de sobrevivir. En el Sewol, ese factor fue Y."

---

## Para la Persona 3 — Dashboard en Power BI

### Cómo cargar el archivo

1. Abre Power BI Desktop.
2. Inicio → Obtener datos → Texto/CSV → selecciona `output/titanic_enriched.csv`.
3. Power BI detectará automáticamente los tipos de columna.

### Ideas para el Dashboard

#### Página 1: Perfil del pasajero superviviente (Titanic)
- KPI cards: Tasa global (38,4%), Mujeres (74,2%), Hombres (18,9%)
- Gráfico de barras: Tasa de supervivencia por Pclass
- Heatmap (matriz): Género × Pclass con `SurvivalProb` como valor
- Filtros: Pclass, Sex, AgeGroup

#### Página 2: Factores de supervivencia
- Gráfico de barras horizontal: Importancia de variables (usar los datos del shap_bar.png o crearlos manualmente)
- Gráfico de dispersión: Edad vs. SurvivalProb, coloreado por Sexo
- Árbol de factores: Pclass → Sex → AgeGroup con tasas de supervivencia

#### Página 3: Comparativa Titanic vs. Sewol (para Persona 2)
- Gráfico doble: Tasa por género en cada desastre
- Narrativa: ¿Cambian los patrones 100 años después?

#### Medidas DAX útiles

```dax
-- Tasa de supervivencia
Survival Rate = AVERAGE(titanic_enriched[Survived])

-- Probabilidad media por grupo
Avg Survival Prob = AVERAGE(titanic_enriched[SurvivalProb])

-- Solo pasajeros de train (datos reales)
Train Survival Rate = 
CALCULATE(
    AVERAGE(titanic_enriched[Survived]),
    titanic_enriched[Split] = "train"
)
```

### Imágenes ya generadas para usar en el informe

Todos los gráficos están en `output/` y pueden insertarse directamente en el Word. Especialmente:
- `eda_heatmap_genero_clase.png` → muy visual para el informe
- `shap_summary.png` → explica la importancia de variables
- `roc_curves.png` → demuestra la calidad del modelo

---

## Resultados clave para el informe (ya calculados)

| Métrica | Valor |
|---------|-------|
| Tasa global de supervivencia | 38,4% |
| Supervivencia mujeres | 74,2% |
| Supervivencia hombres | 18,9% |
| Supervivencia 1ª clase | 63,0% |
| Supervivencia 3ª clase | 24,2% |
| Accuracy del modelo (validación) | **82,1%** |
| ROC-AUC del modelo | **85,6%** |
| CV Accuracy 5-fold | **82,6% ± 2,0%** |
| Variable más importante (SHAP) | Género (Sex_num) |

---

## Preguntas y contacto

Si tenéis dudas sobre el CSV o los resultados, pedidme cualquier aclaración. El notebook `TFE_Titanic_ML.ipynb` documenta cada paso del análisis con explicaciones en markdown — podéis abrirlo en Jupyter para ver exactamente qué se hizo y por qué.

**Próximos pasos del equipo:**
- [ ] Persona 2: Análisis Sewol + comparativa con titanic_enriched.csv
- [ ] Persona 3: Dashboard Power BI + redacción memoria final
- [ ] Albert: Simulador interactivo (en desarrollo)
- [ ] Todos: Revisión conjunta antes del 11 de junio

---

*Guía elaborada por Albert · Mayo 2026*
