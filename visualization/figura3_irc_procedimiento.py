import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_parquet("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos/Nacional/curated/irc_por_nif_ccaa_cpv.parquet",
                     columns=["adjudicatario_key", "irc_base", "flag_B1", "flag_B6"])
df = df.dropna(subset=["irc_base", "flag_B1", "flag_B6"])


def categorizar(row):
    if row["flag_B6"] == 1:
        return "Proc. excepcional\n(oferta única / sin publicidad)"
    elif row["flag_B1"] == 1:
        return "Baja concurrencia\n(≤3 ofertas)"
    else:
        return "Alta concurrencia\n(>3 ofertas)"


df["cat_proc"] = df.apply(categorizar, axis=1)
df = df[df["irc_base"] > 0]

orden = (df.groupby("cat_proc")["irc_base"]
           .median()
           .sort_values(ascending=False)
           .index.tolist())
grupos = [df[df["cat_proc"] == c]["irc_base"].values for c in orden]
ns     = [len(g) for g in grupos]

fig, ax = plt.subplots(figsize=(11, 6))
bp = ax.boxplot(grupos, tick_labels=[f"{c}\n(n={n:,})" for c, n in zip(orden, ns)],
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=2))

colores = ["#d7191c", "#fdae61", "#1a9641"]
for patch, color in zip(bp["boxes"], colores):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

for i, (med, g) in enumerate(zip(bp["medians"], grupos), 1):
    medval = np.median(g)
    ax.text(i, medval + 0.005, f"{medval:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="black")

ax.set_ylabel("IRC (irc_base)", fontsize=11)
ax.set_title(
    "Distribución del IRC por tipo de procedimiento de adjudicación\n"
    r"(Kruskal-Wallis $\eta^2$ = 0,1934 — efecto GRANDE, $\alpha$ < 0,001)",
    fontsize=12, fontweight="bold"
)
ax.tick_params(axis="x", labelsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("/Volumes/Datos Jan/Base De Datos TFG JAN/figura3_irc_procedimiento.png", dpi=300, bbox_inches="tight")
print("Figura 3 guardada.")
