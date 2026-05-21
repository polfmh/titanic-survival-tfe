"""Regenera los 6 gráficos del Bloque II con texto en español."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, 'output')

# ── Carga de datos ─────────────────────────────────────────────────────────
titanic = pd.read_csv(os.path.join(OUTPUT, 'titanic_enriched.csv'))
sewol   = pd.read_csv(os.path.join(BASE, 'sewol', 'sewol_eng.csv'))

titanic_train = titanic[titanic['Split'] == 'train'].copy()

titanic_norm = pd.DataFrame({
    'disaster': 'titanic',
    'gender':   titanic_train['Sex'],
    'age':      titanic_train['Age'],
    'survived': titanic_train['Survived'].astype(float),
    'role':     'passenger',
})

sewol_norm = pd.DataFrame({
    'disaster': 'sewol',
    'gender':   sewol['gender'],
    'age':      sewol['age'],
    'survived': sewol['Raw'].map({'survival': 1, 'Dead': 0}).astype(float),
    'role':     sewol['Category-1'].apply(lambda x: 'crew' if x == 'sailor' else 'passenger'),
})
sewol_norm['age'] = sewol_norm['age'].fillna(sewol_norm['age'].median())

combined   = pd.concat([titanic_norm, sewol_norm], ignore_index=True)
passengers = combined[combined['role'] == 'passenger'].copy()

# ── Estadísticas base ──────────────────────────────────────────────────────
bins         = [0, 13, 20, 40, 60, 100]
age_labels_c = ['Niño (0-12)', 'Adolescente (13-19)', 'Adulto joven (20-39)', 'Adulto (40-59)', 'Mayor (60+)']

results = {}
for disaster in ['titanic', 'sewol']:
    d = passengers[passengers['disaster'] == disaster].copy()
    female_rate = d[d['gender'] == 'female']['survived'].mean() * 100
    male_rate   = d[d['gender'] == 'male']['survived'].mean() * 100
    d['age_group'] = pd.cut(d['age'], bins=bins, labels=age_labels_c, right=False)
    age_rates = {
        f'Supervivencia {g} (%)': round(d[d['age_group'] == g]['survived'].mean() * 100, 1)
        for g in age_labels_c
    }
    results[disaster] = {
        'Tasa global (%)':           round(d['survived'].mean() * 100, 1),
        'Supervivencia mujeres (%)': round(female_rate, 1),
        'Supervivencia hombres (%)': round(male_rate, 1),
        **age_rates,
    }

survival_table = passengers.groupby(['disaster', 'gender']).agg(
    total    = ('survived', 'count'),
    survived = ('survived', 'sum'),
).reset_index()
survival_table['survival_rate'] = (survival_table['survived'] / survival_table['total'] * 100).round(1)

# ── 1. Supervivencia por género ────────────────────────────────────────────
disasters = ['titanic', 'sewol']
genders   = ['female', 'male']
colors    = {'titanic': '#2196F3', 'sewol': '#F44336'}
labels    = {'titanic': 'Titanic (1912)', 'sewol': 'Sewol (2014)'}
x         = np.arange(len(genders))
width     = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
for i, disaster in enumerate(disasters):
    rates = [
        survival_table.loc[
            (survival_table['disaster'] == disaster) & (survival_table['gender'] == g),
            'survival_rate'
        ].values[0]
        for g in genders
    ]
    bars = ax.bar(x + (i - 0.5) * width, rates, width,
                  label=labels[disaster], color=colors[disaster], alpha=0.85)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rate}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Género')
ax.set_ylabel('Tasa de supervivencia (%)')
ax.set_title('Supervivencia por género: Titanic vs Sewol\n(solo pasajeros)')
ax.set_xticks(x)
ax.set_xticklabels(['Mujeres', 'Hombres'], fontsize=12)
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'comparativa_genere_titanic_sewol.png'), dpi=150)
plt.close()
print('OK comparativa_genere_titanic_sewol.png')

# ── 2. Supervivencia por grupo de edad ────────────────────────────────────
passengers_age = passengers.copy()
passengers_age['age_group'] = pd.cut(passengers_age['age'], bins=bins, labels=age_labels_c, right=False)

age_table = passengers_age.groupby(['disaster', 'age_group'], observed=True).agg(
    total    = ('survived', 'count'),
    survived = ('survived', 'sum'),
).reset_index()
age_table['survival_rate'] = (age_table['survived'] / age_table['total'] * 100).round(1)

fig, ax = plt.subplots(figsize=(9, 5))
for disaster, color, label in [('titanic', '#2196F3', 'Titanic (1912)'),
                                ('sewol',   '#F44336', 'Sewol (2014)')]:
    subset = age_table[age_table['disaster'] == disaster]
    ax.plot(subset['age_group'].astype(str), subset['survival_rate'],
            marker='o', linewidth=2.5, markersize=7, color=color, label=label)
    for _, row in subset.iterrows():
        ax.annotate(f"{row['survival_rate']}%",
                    (str(row['age_group']), row['survival_rate']),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=9, color=color)

ax.set_xlabel('Grupo de edad')
ax.set_ylabel('Tasa de supervivencia (%)')
ax.set_title('Supervivencia por grupo de edad: Titanic vs Sewol\n(solo pasajeros)')
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'comparativa_edat_titanic_sewol.png'), dpi=150)
plt.close()
print('OK comparativa_edat_titanic_sewol.png')

# ── 3. Ratio mujeres/hombres ──────────────────────────────────────────────
titanic_ratio = round(results['titanic']['Supervivencia mujeres (%)'] /
                      results['titanic']['Supervivencia hombres (%)'], 2)
sewol_ratio   = round(results['sewol']['Supervivencia mujeres (%)'] /
                      results['sewol']['Supervivencia hombres (%)'], 2)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['Titanic (1912)', 'Sewol (2014)'], [titanic_ratio, sewol_ratio],
               color=['#2196F3', '#F44336'], alpha=0.85, width=0.4)
ax.axhline(y=1, color='black', linestyle='--', linewidth=1.2, label='Ratio = 1 (sin diferencia de género)')
for bar, val in zip(bars, [titanic_ratio, sewol_ratio]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'{val}x', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Ratio supervivencia mujeres / hombres')
ax.set_title('Impacto del género en la supervivencia\nTitanic vs Sewol')
ax.set_ylim(0, titanic_ratio + 0.5)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'ratio_genere_titanic_sewol.png'), dpi=150)
plt.close()
print('OK ratio_genere_titanic_sewol.png')

# ── 4. Niños vs adultos ───────────────────────────────────────────────────
categories     = ['Niños (0-12)', 'Adultos (13+)']
titanic_rates_ch = [
    passengers[(passengers['disaster'] == 'titanic') & (passengers['age'] < 13)]['survived'].mean() * 100,
    passengers[(passengers['disaster'] == 'titanic') & (passengers['age'] >= 13)]['survived'].mean() * 100,
]
sewol_rates_ch = [
    passengers[(passengers['disaster'] == 'sewol') & (passengers['age'] < 13)]['survived'].mean() * 100,
    passengers[(passengers['disaster'] == 'sewol') & (passengers['age'] >= 13)]['survived'].mean() * 100,
]

x     = np.arange(len(categories))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 5))
bars1 = ax.bar(x - width/2, titanic_rates_ch, width, label='Titanic (1912)', color='#2196F3', alpha=0.85)
bars2 = ax.bar(x + width/2, sewol_rates_ch,   width, label='Sewol (2014)',   color='#F44336', alpha=0.85)
for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Tasa de supervivencia (%)')
ax.set_title('Supervivencia niños vs adultos: Titanic vs Sewol\n(solo pasajeros)')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'ratio_nens_adults_titanic_sewol.png'), dpi=150)
plt.close()
print('OK ratio_nens_adults_titanic_sewol.png')

# ── 5. Barras horizontales comparativa ────────────────────────────────────
indicadors = [
    'Tasa global (%)', 'Supervivencia mujeres (%)', 'Supervivencia hombres (%)',
    'Supervivencia Niño (0-12) (%)', 'Supervivencia Adolescente (13-19) (%)',
    'Supervivencia Adulto joven (20-39) (%)', 'Supervivencia Adulto (40-59) (%)',
    'Supervivencia Mayor (60+) (%)',
]
etiquetas = [
    'Tasa global', 'Mujeres', 'Hombres',
    'Niños (0-12)', 'Adolescentes (13-19)',
    'Adulto joven (20-39)', 'Adulto (40-59)', 'Mayor (60+)',
]
titanic_vals = [results['titanic'].get(i, 0) for i in indicadors]
sewol_vals   = [results['sewol'].get(i, 0)   for i in indicadors]

y      = np.arange(len(etiquetas))
height = 0.35
fig, ax = plt.subplots(figsize=(10, 7))
bars1 = ax.barh(y + height/2, titanic_vals, height, label='Titanic (1912)', color='#2196F3', alpha=0.85)
bars2 = ax.barh(y - height/2, sewol_vals,   height, label='Sewol (2014)',   color='#F44336', alpha=0.85)
for bar, val in zip(bars1, titanic_vals):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{val}%', va='center', fontsize=9, fontweight='bold', color='#2196F3')
for bar, val in zip(bars2, sewol_vals):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{val}%', va='center', fontsize=9, fontweight='bold', color='#F44336')
ax.set_xlabel('Tasa de supervivencia (%)')
ax.set_title('Comparativa de supervivencia: Titanic vs Sewol')
ax.set_yticks(y)
ax.set_yticklabels(etiquetas, fontsize=11)
ax.set_xlim(0, 120)
ax.legend()
ax.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'barres_comparativa_titanic_sewol.png'), dpi=150)
plt.close()
print('OK barres_comparativa_titanic_sewol.png')

# ── 6. Radar ──────────────────────────────────────────────────────────────
cat_radar = ['Mujeres', 'Tasa global', 'Adulto\n(40-59)', 'Adulto joven\n(20-39)', 'Adolescentes', 'Hombres']
N = len(cat_radar)

titanic_v = [
    results['titanic']['Supervivencia mujeres (%)'],
    results['titanic']['Tasa global (%)'],
    results['titanic']['Supervivencia Adulto (40-59) (%)'],
    results['titanic']['Supervivencia Adulto joven (20-39) (%)'],
    results['titanic']['Supervivencia Adolescente (13-19) (%)'],
    results['titanic']['Supervivencia hombres (%)'],
]
sewol_v = [
    results['sewol']['Supervivencia mujeres (%)'],
    results['sewol']['Tasa global (%)'],
    results['sewol']['Supervivencia Adulto (40-59) (%)'],
    results['sewol']['Supervivencia Adulto joven (20-39) (%)'],
    results['sewol']['Supervivencia Adolescente (13-19) (%)'],
    results['sewol']['Supervivencia hombres (%)'],
]

angles    = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
titanic_c = titanic_v + titanic_v[:1]
sewol_c   = sewol_v   + sewol_v[:1]
angles_c  = angles    + angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles_c, titanic_c, 'o-', linewidth=2, color='#2196F3', label='Titanic (1912)')
ax.fill(angles_c, titanic_c, alpha=0.15, color='#2196F3')
ax.plot(angles_c, sewol_c, 'o-', linewidth=2, color='#F44336', label='Sewol (2014)')
ax.fill(angles_c, sewol_c, alpha=0.15, color='#F44336')
ax.set_thetagrids(np.degrees(angles), cat_radar, fontsize=12)
ax.set_ylim(0, 100)
ax.set_title('Perfil de supervivencia: Titanic vs Sewol', size=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'radar_titanic_sewol.png'), dpi=150)
plt.close()
print('OK radar_titanic_sewol.png')

print('\nTodos los graficos regenerados en espanol.')
