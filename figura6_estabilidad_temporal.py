import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

NAMES = {
    "AN": "Andalucía", "GA": "Galicia", "NAC": "Nacional (PLACSP)", "MD": "Madrid",
    "IB": "Baleares", "EX": "Extremadura", "RI": "La Rioja", "MU": "Murcia",
    "VC": "C. Valenciana", "CT": "Cataluña", "CN": "Canarias", "CE": "Ceuta",
    "AR": "Aragón", "CB": "Cantabria", "ML": "Melilla", "CM": "Castilla-La Mancha",
    "NA": "Navarra", "CL": "Castilla y León", "PV": "País Vasco", "AS": "Asturias"
}

df = pd.read_csv("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos/Nacional/outputs/mapas/M05_irc_estabilidad_temporal.csv")
df = df.dropna(subset=["ccaa"])
df["nombre"] = df["ccaa"].map(NAMES).fillna(df["ccaa_nombre"])
df["rank_19_21"] = df["irc_media_2019_2021"].rank(ascending=False).astype(int)
df["rank_22_25"] = df["irc_media_2022_2025"].rank(ascending=False).astype(int)

rho, pval = spearmanr(df["rank_19_21"], df["rank_22_25"])

fig, ax = plt.subplots(figsize=(9, 8))
colors = ["#d7191c" if d > 0 else "#1a9641" for d in df["delta_irc"]]
ax.scatter(df["rank_19_21"], df["rank_22_25"], s=120, c=colors, zorder=5,
           edgecolors="white", linewidths=0.5)

for _, row in df.iterrows():
    ax.annotate(row["nombre"], (row["rank_19_21"], row["rank_22_25"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

n = len(df)
ax.plot([1, n], [1, n], "--", color="gray", linewidth=1, alpha=0.7)

ax.set_xlabel("Ranking IRC — subperíodo 2019–2021\n(1 = mayor riesgo)", fontsize=11)
ax.set_ylabel("Ranking IRC — subperíodo 2022–2025\n(1 = mayor riesgo)", fontsize=11)
ax.set_title(f"Estabilidad temporal del IRC territorial\n"
             f"ρ de Spearman = {rho:.4f}  (umbral robustez ≥ 0,70)",
             fontsize=12, fontweight="bold")
ax.set_xticks(range(1, n + 1))
ax.set_yticks(range(1, n + 1))
ax.invert_xaxis()
ax.invert_yaxis()
ax.grid(alpha=0.3)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d7191c', markersize=9, label='IRC aumentó (empeoró)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1a9641', markersize=9, label='IRC disminuyó (mejoró)'),
    Line2D([0], [0], linestyle='--', color='gray', label='Concordancia perfecta (ρ=1)'),
]
ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("/Volumes/Datos Jan/Base De Datos TFG JAN/figura6_estabilidad_temporal.png", dpi=300, bbox_inches="tight")
print("Figura 6 guardada.")
