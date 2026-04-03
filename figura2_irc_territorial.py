import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

CCAA_NAMES = {
    "AN": "Andalucía", "GA": "Galicia", "NAC": "Nacional (PLACSP)",
    "MD": "Madrid", "IB": "Baleares", "EX": "Extremadura",
    "RI": "La Rioja", "MU": "Murcia", "VC": "C. Valenciana",
    "CT": "Cataluña", "CN": "Canarias", "CE": "Ceuta",
    "AR": "Aragón", "CB": "Cantabria", "ML": "Melilla",
    "CM": "Castilla-La Mancha", "NA": "Navarra",
    "CL": "Castilla y León", "PV": "País Vasco", "AS": "Asturias"
}

df = pd.read_parquet("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos/Nacional/curated/irc_agregado_territorial.parquet")
df["nombre"] = df["ccaa"].map(CCAA_NAMES)
df["importe_Md"] = df["importe_total_ccaa"] / 1e9
df = df.sort_values("irc_mediana_pond", ascending=True)

norm = mcolors.Normalize(vmin=df["irc_mediana_pond"].min() - 0.02, vmax=df["irc_mediana_pond"].max() + 0.02)
cmap = plt.cm.RdYlGn_r
colors = [cmap(norm(v)) for v in df["irc_mediana_pond"]]

fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.barh(df["nombre"], df["irc_mediana_pond"], color=colors, edgecolor="white", linewidth=0.5)

for bar, val, imp in zip(bars, df["irc_mediana_pond"], df["importe_Md"]):
    label = f"{val:.2f}"
    if imp >= 1:
        label += f"  ({imp:.0f} Md€)"
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            label, va="center", ha="left", fontsize=9, fontweight="bold")

irc_nacional = df.loc[df["ccaa"] == "NAC", "irc_mediana_pond"].values[0]
ax.axvline(x=irc_nacional, color="gray", linestyle="--", linewidth=1, alpha=0.8,
           label=f"IRC Nacional ({irc_nacional:.2f})")
ax.set_xlabel("IRC agregado (mediana ponderada por importe)", fontsize=11)
ax.set_title("Índice de Riesgo de Contratación por territorio\n(período 2019–2025)",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, df["irc_mediana_pond"].max() + 0.12)
ax.legend(fontsize=9)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
cbar.set_label("Nivel de IRC", fontsize=9)

plt.tight_layout()
plt.savefig("/Volumes/Datos Jan/Base De Datos TFG JAN/figura2_irc_territorial.png", dpi=300, bbox_inches="tight")
print("Figura 2 guardada.")
