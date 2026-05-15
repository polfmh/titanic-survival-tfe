# TFE — Supervivencia en Desastres Marítimos: del Titanic al Sewol
**Posgrado en Business Analytics (PBA) · UPC School · 3ª Edición · 2026**

Pablo Ferrer-Mayol · Roger Tarridas · Albert Felis

---

## Estructura del proyecto

```
tfg_titanic/
│
├── titanic_kaggle/          ← Datos originales de la competición Kaggle
│   ├── train.csv            ← 891 pasajeros con etiqueta de supervivencia
│   ├── test.csv             ← 418 pasajeros para el submission
│   └── gender_submission.csv← Ejemplo de formato de entrega
│
├── sewol/                   ← Dataset del ferry MV Sewol
│   └── sewol_eng.csv        ← Datos del naufragio de 2014
│
├── notebooks/               ← Análisis completo en Jupyter
│   └── TFE_Titanic_ML.ipynb ← Notebook principal: EDA + Feature Engineering + ML
│
├── output/                  ← Resultados del modelo y gráficos
│   ├── titanic_submission.csv   ← Predicciones para Kaggle (418 filas)
│   ├── titanic_enriched.csv     ← Dataset completo con probabilidades (1309 filas)
│   ├── eda_*.png                ← Gráficos del análisis exploratorio
│   ├── shap_*.png               ← Gráficos de importancia de variables (SHAP)
│   ├── confusion_matrices.png   ← Matrices de confusión de los modelos
│   ├── roc_curves.png           ← Curvas ROC comparativas
│   └── feature_importance_rf.png← Importancia de variables del Random Forest
│
├── simulator/               ← Herramienta interactiva web
│   ├── index.html           ← Simulador: "¿Habrías sobrevivido al Titanic?"
│   └── model_explorer.html  ← Dashboard académico del modelo
│
├── docs/                    ← Documentación del proyecto
│   ├── informe_bloque_ML_Albert.md  ← Sección del informe TFE (Albert)
│   ├── GUIA_COMPANEROS.md           ← Instrucciones para el equipo
│   ├── UPDATE_EQUIPO.md             ← Resumen de avance del equipo
│   └── CHEATSHEET_DEFENSA.md        ← Guía para la presentación al tribunal
│
├── scripts/                 ← Scripts de utilidad (uso interno)
│   ├── download_titanic.py  ← Descarga del dataset de Kaggle
│   └── create_notebook.py   ← Generador del notebook Jupyter
│
├── requirements.txt         ← Dependencias Python del proyecto
└── README.md                ← Este archivo
```

---

## Cómo abrir el simulador

Abre directamente en el navegador:
- **Simulador:** `simulator/index.html`
- **Model Explorer:** `simulator/model_explorer.html`

No requiere servidor ni conexión a internet (excepto para cargar las fuentes de Google Fonts).

---

## Cómo ejecutar el notebook

```bash
pip install -r requirements.txt
jupyter notebook notebooks/TFE_Titanic_ML.ipynb
```

---

## Resultados clave del modelo

| Métrica | Valor |
|---------|-------|
| Precisión (validación) | 82,1% |
| ROC-AUC | 85,6% |
| Precisión (val. cruzada) | 82,6% ± 2,0% |
| Variable más importante | Género (28,7%) |
| Supervivencia mujeres | 74,2% |
| Supervivencia hombres | 18,9% |

---

## Entregables finales

- `output/titanic_submission.csv` → submission para Kaggle
- `output/titanic_enriched.csv` → CSV para el equipo (Power BI, comparativa Sewol)
- `simulator/index.html` → simulador interactivo
- `simulator/model_explorer.html` → dashboard académico
- `notebooks/TFE_Titanic_ML.ipynb` → notebook completo documentado
- `docs/informe_bloque_ML_Albert.md` → sección del informe TFE
