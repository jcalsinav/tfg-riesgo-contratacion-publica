import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_parquet("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos/Nacional/curated/irc_por_nif_ccaa_cpv.parquet")

irc = df["irc_base"].dropna()
irc_nonzero = irc[irc > 0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribución del Índice de Riesgo de Contratación (IRC)", fontsize=14, fontweight="bold")

axes[0].hist(irc, bins=100, color="#2c7bb6", edgecolor="white", linewidth=0.3)
axes[0].set_title("Distribución completa (incluye score = 0)")
axes[0].set_xlabel("IRC (irc_base)")
axes[0].set_ylabel("Número de grupos")
axes[0].annotate(
    f"Score = 0: {(irc == 0).mean()*100:.1f}%\nN = {len(irc):,}",
    xy=(0.65, 0.80), xycoords="axes fraction",
    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray")
)

axes[1].hist(irc_nonzero, bins=80, color="#d7191c", edgecolor="white", linewidth=0.3)
axes[1].set_yscale("log")
axes[1].set_title("Grupos con IRC > 0 (escala logarítmica en Y)")
axes[1].set_xlabel("IRC (irc_base)")
axes[1].set_ylabel("Número de grupos (log)")
axes[1].annotate(
    f"P75 = {irc.quantile(0.75):.3f}\nMax = {irc.max():.3f}\nMedia = {irc.mean():.4f}",
    xy=(0.55, 0.75), xycoords="axes fraction",
    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray")
)

plt.tight_layout()
plt.savefig("/Volumes/Datos Jan/Base De Datos TFG JAN/figura1_distribucion_irc.png", dpi=300, bbox_inches="tight")
print("Figura 1 guardada.")
