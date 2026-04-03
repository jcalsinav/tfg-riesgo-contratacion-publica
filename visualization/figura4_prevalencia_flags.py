import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_parquet("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos/Nacional/curated/irc_por_nif_ccaa_cpv.parquet")

df_nif = df.groupby("adjudicatario_key")[["F1", "F2", "F3", "F4", "F5", "borme_cobertura"]].max().reset_index()
n_total = len(df_nif)
n_borme = (df_nif["borme_cobertura"] > 0).sum()
df_borme = df_nif[df_nif["borme_cobertura"] > 0]

corp_labels = ["F1\nConstitución\nreciente", "F2\nCapital social\nbajo",
               "F3\nAdministradores\ncompartidos", "F4\nDisolución\npost-adj.",
               "F5\nSituación\nconcursal"]
corp_total = [(df_nif[f] == 1).mean() * 100 for f in ["F1", "F2", "F3", "F4", "F5"]]
corp_borme = [(df_borme[f] == 1).mean() * 100 for f in ["F1", "F2", "F3", "F4", "F5"]]

n_grupos = len(df)
comp_labels = ["B1\nOferta\núnica", "B2\nConc.\nHHI", "B3\nRacha\nadj.",
               "B4\nClustering\nmenor", "B5\nClustering\nSARA", "B6\nSin\npublicidad"]
comp_vals = [(df[f] == 1).mean() * 100 for f in ["flag_B1", "flag_B2", "flag_B3", "flag_B4", "flag_B5", "flag_B6"]]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Prevalencia de indicadores de riesgo en el corpus analítico (2019–2025)",
             fontsize=13, fontweight="bold")

x = np.arange(len(corp_labels))
w = 0.35
b1 = axes[0].bar(x - w/2, corp_total, width=w, label="Corpus total",
                  color="#2c7bb6", alpha=0.85, edgecolor="white")
b2 = axes[0].bar(x + w/2, corp_borme, width=w, label=f"Subconjunto BORME (n={n_borme:,})",
                  color="#d7191c", alpha=0.85, edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels(corp_labels, fontsize=9)
axes[0].set_ylabel("% adjudicatarios con flag activo")
axes[0].set_title(f"Indicadores corporativos (F1–F5)\nN adjudicatarios únicos = {n_total:,}")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h > 0.3:
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.2,
                     f"{h:.2f}%", ha="center", va="bottom", fontsize=8)

colors = ["#d7191c" if v > 10 else "#fdae61" if v > 3 else "#1a9641" for v in comp_vals]
axes[1].bar(range(len(comp_labels)), comp_vals, color=colors, alpha=0.85, edgecolor="white")
axes[1].set_xticks(range(len(comp_labels)))
axes[1].set_xticklabels(comp_labels, fontsize=9)
axes[1].set_ylabel("% grupos con flag activo")
axes[1].set_title(f"Indicadores de comportamiento (B1–B6)\nN grupos = {n_grupos:,}")
axes[1].grid(axis="y", alpha=0.3)
for i, v in enumerate(comp_vals):
    axes[1].text(i, v + 0.2, f"{v:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("/Volumes/Datos Jan/Base De Datos TFG JAN/figura4_prevalencia_flags.png", dpi=300, bbox_inches="tight")
print("Figura 4 guardada.")
