import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_parquet("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos/Nacional/curated/irc_por_nif_ccaa_cpv.parquet",
                     columns=["irc_base", "importe_total"])
df = df[(df["irc_base"] > 0) & (df["importe_total"] > 0)].copy()
df["log_importe"] = np.log(df["importe_total"])

sample = df[["irc_base", "log_importe"]].dropna().sample(min(15000, len(df)), random_state=42)
slope, intercept, r, p, se = stats.linregress(sample["log_importe"], sample["irc_base"])
r2 = r**2

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(sample["log_importe"], sample["irc_base"],
           alpha=0.15, s=8, color="#2c7bb6", rasterized=True)

x_line = np.linspace(sample["log_importe"].min(), sample["log_importe"].max(), 200)
ax.plot(x_line, intercept + slope * x_line, color="#d7191c", linewidth=2,
        label=f"Regresión lineal (β = {slope:.5f}, R² = {r2:.4f})")

ax.set_xlabel("log(Importe adjudicación)", fontsize=11)
ax.set_ylabel("IRC (irc_base)", fontsize=11)
ax.set_title(
    "Relación entre importe adjudicado e IRC\n"
    "(β = 0,00961, t = 104,76, p < 0,001 — modelo MCO con efectos fijos CCAA×CPV)",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.annotate("Nota: scatter sobre muestra aleatoria de 15.000 grupos (IRC > 0).\n"
            "β del título corresponde al modelo MCO con efectos fijos (Script 07).",
            xy=(0.02, 0.96), xycoords="axes fraction", fontsize=7.5,
            va="top", color="gray")

plt.tight_layout()
plt.savefig("/Volumes/Datos Jan/Base De Datos TFG JAN/figura5_scatter_irc_importe.png", dpi=300, bbox_inches="tight")
print("Figura 5 guardada.")
