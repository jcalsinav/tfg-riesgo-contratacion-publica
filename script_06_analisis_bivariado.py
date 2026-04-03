"""
script_06_analisis_bivariado.py
================================
Análisis de relaciones bivariadas entre el IRC y las variables
de contexto: procedimiento, importe y competencia.

Apartados del TFG que alimenta: 5.d, 5.e, 5.f, 5.g

Pruebas estadísticas
--------------------
Variables continuas  → Correlación de Spearman + tamaño de efecto r
Variables categóricas → Kruskal-Wallis + corrección Bonferroni + η²
Concentración por órgano → HHI + top-5 share por (CCAA, CPV2d)

Inputs
------
  Nacional/curated/irc_por_nif_ccaa_cpv.parquet         (Script 03)
  Nacional/curated/indicadores_comportamiento_raw.parquet (Script 02)

Outputs  (Nacional/outputs/bivariado/)
-------
  B01_correlacion_irc_importe.csv
  B02_correlacion_irc_num_ofertas.csv
  B03_kruskal_irc_procedimiento.csv
  B04_kruskal_irc_tipo_contrato.csv
  B05_kruskal_irc_ccaa.csv
  B06_posthoc_procedimiento.csv          ← comparaciones por pares
  B07_concentracion_organo_ccaa.csv
  B08_irc_vs_oferta_unica_por_ccaa.csv
  bivariado_quality.txt
  logs/script_06.log
"""

from __future__ import annotations
from pathlib import Path
import logging
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

# ── RUTAS ─────────────────────────────────────────────────────────────────────
BASE = Path("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos")

IRC_PATH = BASE / "Nacional/curated/irc_por_nif_ccaa_cpv.parquet"
RAW_PATH = BASE / "Nacional/curated/indicadores_comportamiento_raw.parquet"

OUT_DIR     = BASE / "Nacional/outputs/bivariado"
REPORTS_DIR = BASE / "Nacional/reports"
LOGS_DIR    = BASE / "Nacional/logs"
for d in [OUT_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

log_path = LOGS_DIR / "script_06.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

ALPHA = 0.05   # nivel de significación


# ── UTILIDADES ────────────────────────────────────────────────────────────────
def guardar(df: pd.DataFrame, nombre: str, desc: str) -> Path:
    ruta = OUT_DIR / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    log.info("  [OK] %s  (%s filas) — %s", nombre, f"{len(df):,}", desc)
    return ruta


def spearman_con_efecto(x: pd.Series, y: pd.Series,
                         etiqueta_x: str, etiqueta_y: str) -> dict:
    """
    Correlación de Spearman entre x e y con tamaño de efecto r.
    r = rho (Spearman rho equivale a r de rango).
    """
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n  = len(df)
    if n < 10:
        return {"var_x": etiqueta_x, "var_y": etiqueta_y,
                "n": n, "rho": np.nan, "p_valor": np.nan,
                "efecto_r": np.nan, "sig": "n/a"}
    rho, p = stats.spearmanr(df["x"], df["y"])
    return {
        "var_x":    etiqueta_x,
        "var_y":    etiqueta_y,
        "n":        n,
        "rho":      round(rho, 4),
        "p_valor":  round(p, 6),
        "efecto_r": round(abs(rho), 4),   # r de rango = |rho|
        "sig":      "Sí" if p < ALPHA else "No",
        "interpretacion": (
            "grande" if abs(rho) >= 0.5
            else "moderado" if abs(rho) >= 0.3
            else "pequeño" if abs(rho) >= 0.1
            else "trivial"
        ),
    }


def kruskal_wallis(irc: pd.Series, grupos: pd.Series,
                    nombre_grupo: str) -> dict:
    """
    Prueba de Kruskal-Wallis + η² (tamaño de efecto).
    η² = (H - k + 1) / (n - k)  donde k = nº grupos, H = estadístico KW.
    """
    df = pd.DataFrame({"irc": irc, "g": grupos}).dropna()
    cats = df["g"].unique()
    k    = len(cats)
    n    = len(df)

    if k < 2 or n < 10:
        return {"variable": nombre_grupo, "k_grupos": k, "n": n,
                "H": np.nan, "p_valor": np.nan, "eta2": np.nan, "sig": "n/a"}

    muestras = [df.loc[df["g"] == c, "irc"].values for c in cats]
    H, p = stats.kruskal(*muestras)

    eta2 = max((H - k + 1) / (n - k), 0) if n > k else np.nan

    return {
        "variable":   nombre_grupo,
        "k_grupos":   k,
        "n":          n,
        "H":          round(H, 4),
        "p_valor":    round(p, 6),
        "eta2":       round(eta2, 4),
        "sig":        "Sí" if p < ALPHA else "No",
        "interpretacion": (
            "grande" if eta2 >= 0.14
            else "moderado" if eta2 >= 0.06
            else "pequeño" if eta2 >= 0.01
            else "trivial"
        ) if not np.isnan(eta2) else "n/a",
    }


def posthoc_bonferroni(irc: pd.Series, grupos: pd.Series,
                        nombre_grupo: str) -> pd.DataFrame:
    """
    Comparaciones Mann-Whitney por pares con corrección de Bonferroni.
    Solo se aplica si el KW global fue significativo.
    """
    df  = pd.DataFrame({"irc": irc, "g": grupos}).dropna()
    cats = sorted(df["g"].unique())
    pares = list(combinations(cats, 2))
    m    = len(pares)   # nº comparaciones para Bonferroni

    rows = []
    for a, b in pares:
        xa = df.loc[df["g"] == a, "irc"].values
        xb = df.loc[df["g"] == b, "irc"].values
        if len(xa) < 3 or len(xb) < 3:
            continue
        U, p_raw = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        p_adj = min(p_raw * m, 1.0)   # Bonferroni
        # r de efecto = U / (n1 * n2)  → normalizado a [0,1]
        r = U / (len(xa) * len(xb))
        rows.append({
            "variable":       nombre_grupo,
            "grupo_A":        a,
            "grupo_B":        b,
            "mediana_A":      round(df.loc[df["g"] == a, "irc"].median(), 4),
            "mediana_B":      round(df.loc[df["g"] == b, "irc"].median(), 4),
            "U":              round(U, 2),
            "p_raw":          round(p_raw, 6),
            "p_bonferroni":   round(p_adj, 6),
            "efecto_r":       round(abs(r - 0.5) * 2, 4),  # centrado en 0
            "sig_bonferroni": "Sí" if p_adj < ALPHA else "No",
        })

    return pd.DataFrame(rows)


# ── B01: IRC vs log(importe) ───────────────────────────────────────────────────
def tabla_B01(irc: pd.DataFrame) -> pd.DataFrame:
    log.info("B01: correlación IRC vs log(importe)...")

    sub = irc[irc["irc_base"].notna() & irc.get("importe_total", pd.Series(dtype=float)).notna()].copy()
    if "importe_total" not in sub.columns:
        log.warning("  B01: importe_total no disponible.")
        return pd.DataFrame()

    sub = sub[sub["importe_total"] > 0].copy()
    sub["log_importe"] = np.log(sub["importe_total"])

    rows = []

    # Global
    rows.append(spearman_con_efecto(
        sub["irc_base"], sub["log_importe"],
        "irc_base", "log_importe_global"
    ))

    # Por CCAA
    for ccaa, g in sub.groupby("ccaa"):
        r = spearman_con_efecto(g["irc_base"], g["log_importe"],
                                 "irc_base", f"log_importe_{ccaa}")
        r["ccaa"] = ccaa
        rows.append(r)

    return pd.DataFrame(rows)


# ── B02: IRC vs num_ofertas ────────────────────────────────────────────────────
def tabla_B02(irc: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    log.info("B02: correlación IRC vs num_ofertas (a través de B1_ratio)...")

    if "B1_ratio" not in raw.columns:
        log.warning("  B02: B1_ratio no disponible en raw.")
        return pd.DataFrame()

    # Unir IRC base con B1_ratio (proxy de num_ofertas)
    merged = irc[["adjudicatario_key","ccaa","cpv2d","irc_base"]].merge(
        raw[["adjudicatario_key","ccaa","cpv2d","B1_ratio"]],
        on=["adjudicatario_key","ccaa","cpv2d"],
        how="inner",
    ).dropna(subset=["irc_base","B1_ratio"])

    rows = []
    rows.append(spearman_con_efecto(
        merged["irc_base"], merged["B1_ratio"],
        "irc_base", "B1_ratio_oferta_unica_global"
    ))
    for ccaa, g in merged.groupby("ccaa"):
        r = spearman_con_efecto(g["irc_base"], g["B1_ratio"],
                                 "irc_base", f"B1_ratio_{ccaa}")
        r["ccaa"] = ccaa
        rows.append(r)

    return pd.DataFrame(rows)


# ── B03: IRC vs procedimiento ─────────────────────────────────────────────────
def tabla_B03_posthoc(irc: pd.DataFrame, raw: pd.DataFrame):
    log.info("B03: Kruskal-Wallis IRC vs procedimiento...")

    if "B6_ratio" not in raw.columns:
        log.warning("  B03: B6_ratio no disponible.")
        return pd.DataFrame(), pd.DataFrame()

    # Usar flag_B1 como proxy de procedimiento abierto vs cerrado
    # y clasificar grupos por tipo predominante
    merged = irc[["adjudicatario_key","ccaa","cpv2d","irc_base",
                   "flag_B1","flag_B6"]].copy().dropna(subset=["irc_base"])

    # Crear categoría de procedimiento basada en flags
    def categorizar(row):
        if row.get("flag_B6") == 1:
            return "Proc. excepcional"
        elif row.get("flag_B1") == 1:
            return "Baja concurrencia"
        elif row.get("flag_B1") == 0:
            return "Alta concurrencia"
        else:
            return "Sin dato"

    merged["cat_proc"] = merged.apply(categorizar, axis=1)
    merged = merged[merged["cat_proc"] != "Sin dato"]

    kw   = kruskal_wallis(merged["irc_base"], merged["cat_proc"], "Tipo procedimiento")
    posthoc = pd.DataFrame()
    if kw["sig"] == "Sí":
        posthoc = posthoc_bonferroni(merged["irc_base"], merged["cat_proc"],
                                      "Tipo procedimiento")
    return pd.DataFrame([kw]), posthoc


# ── B04: IRC vs tipo de contrato ──────────────────────────────────────────────
def tabla_B04(irc: pd.DataFrame) -> pd.DataFrame:
    log.info("B04: Kruskal-Wallis IRC vs tipo contrato (menor vs no menor)...")

    # Proxy: si el grupo tiene más del 50% de menores → "menor"
    # Usamos flag_B4 como indicador de menores
    sub = irc[irc["irc_base"].notna()].copy()
    if "flag_B4" not in sub.columns:
        log.warning("  B04: flag_B4 no disponible.")
        return pd.DataFrame()

    sub["tipo_cat"] = sub["flag_B4"].map(
        {1.0: "Alto clustering menor", 0.0: "Bajo clustering menor"}
    ).fillna("Sin dato")
    sub = sub[sub["tipo_cat"] != "Sin dato"]

    kw = kruskal_wallis(sub["irc_base"], sub["tipo_cat"], "Clustering contrato menor")
    return pd.DataFrame([kw])


# ── B05: IRC vs CCAA ──────────────────────────────────────────────────────────
def tabla_B05_posthoc(irc: pd.DataFrame):
    log.info("B05: Kruskal-Wallis IRC vs CCAA...")

    sub = irc[irc["irc_base"].notna()].copy()
    kw  = kruskal_wallis(sub["irc_base"], sub["ccaa"], "CCAA")

    posthoc = pd.DataFrame()
    if kw["sig"] == "Sí":
        posthoc = posthoc_bonferroni(sub["irc_base"], sub["ccaa"], "CCAA")

    return pd.DataFrame([kw]), posthoc


# ── B07: Concentración por órgano × CCAA ─────────────────────────────────────
def tabla_B07(irc: pd.DataFrame) -> pd.DataFrame:
    log.info("B07: concentración por órgano × CCAA...")

    if "B2_hhi_medio" not in irc.columns:
        # Derivar de flag_B2 si está disponible
        if "flag_B2" in irc.columns:
            sub = irc[irc["irc_base"].notna()].copy()
            t = sub.groupby("ccaa").agg(
                pct_hhi_alto =("flag_B2","mean"),
                n_grupos     =("adjudicatario_key","count"),
            ).reset_index()
            t["pct_hhi_alto"] = (t["pct_hhi_alto"] * 100).round(2)
            return t
        log.warning("  B07: B2_hhi_medio no disponible.")
        return pd.DataFrame()

    # Cargar raw para tener B2 continuo
    raw_path = BASE / "Nacional/curated/indicadores_comportamiento_raw.parquet"
    if not raw_path.exists():
        return pd.DataFrame()

    raw = pd.read_parquet(raw_path, columns=["adjudicatario_key","ccaa","cpv2d","B2_hhi_medio"])
    merged = irc[["adjudicatario_key","ccaa","cpv2d","irc_base"]].merge(
        raw, on=["adjudicatario_key","ccaa","cpv2d"], how="inner"
    ).dropna(subset=["irc_base","B2_hhi_medio"])

    t07 = merged.groupby("ccaa").agg(
        hhi_mediana  =("B2_hhi_medio","median"),
        hhi_media    =("B2_hhi_medio","mean"),
        hhi_p75      =("B2_hhi_medio",lambda x: x.quantile(0.75)),
        n_grupos     =("adjudicatario_key","count"),
        corr_irc_hhi =(
            "B2_hhi_medio",
            lambda x: x.corr(
                merged.loc[x.index,"irc_base"], method="spearman"
            )
        ),
    ).reset_index()

    return t07.sort_values("hhi_mediana", ascending=False)


# ── B08: IRC vs oferta única por CCAA ─────────────────────────────────────────
def tabla_B08(irc: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    log.info("B08: IRC vs tasa oferta única por CCAA...")

    if "B1_ratio" not in raw.columns:
        return pd.DataFrame()

    merged = irc[["adjudicatario_key","ccaa","cpv2d","irc_base"]].merge(
        raw[["adjudicatario_key","ccaa","cpv2d","B1_ratio","B1_n"]],
        on=["adjudicatario_key","ccaa","cpv2d"], how="inner"
    ).dropna(subset=["irc_base","B1_ratio"])

    rows = []
    for ccaa, g in merged.groupby("ccaa"):
        rows.append({
            "ccaa":              ccaa,
            "n":                 len(g),
            "B1_mediana":        round(g["B1_ratio"].median(), 4),
            "B1_pct_mayor50":    round(100 * (g["B1_ratio"] > 0.5).mean(), 1),
            "irc_mediana":       round(g["irc_base"].median(), 4),
            "corr_spearman":     round(
                g["irc_base"].corr(g["B1_ratio"], method="spearman"), 4
            ),
        })

    return pd.DataFrame(rows).sort_values("B1_mediana", ascending=False)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("SCRIPT 06 — Análisis bivariado")
    log.info("Inicio: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    if not IRC_PATH.exists():
        raise FileNotFoundError(f"IRC no encontrado: {IRC_PATH}")

    irc = pd.read_parquet(IRC_PATH)
    raw = pd.read_parquet(RAW_PATH) if RAW_PATH.exists() else pd.DataFrame()

    log.info("IRC: %s filas | Raw: %s filas", f"{len(irc):,}", f"{len(raw):,}")

    archivos = []

    b01 = tabla_B01(irc)
    if len(b01) > 0:
        archivos.append(guardar(b01, "B01_correlacion_irc_importe.csv",
                                 "Correlación IRC vs log(importe)"))

    b02 = tabla_B02(irc, raw)
    if len(b02) > 0:
        archivos.append(guardar(b02, "B02_correlacion_irc_num_ofertas.csv",
                                 "Correlación IRC vs tasa oferta única"))

    b03, b06 = tabla_B03_posthoc(irc, raw)
    if len(b03) > 0:
        archivos.append(guardar(b03, "B03_kruskal_irc_procedimiento.csv",
                                 "Kruskal-Wallis IRC vs procedimiento"))
    if len(b06) > 0:
        archivos.append(guardar(b06, "B06_posthoc_procedimiento.csv",
                                 "Post-hoc Bonferroni procedimiento"))

    b04 = tabla_B04(irc)
    if len(b04) > 0:
        archivos.append(guardar(b04, "B04_kruskal_irc_tipo_contrato.csv",
                                 "Kruskal-Wallis IRC vs tipo contrato"))

    b05, b05_post = tabla_B05_posthoc(irc)
    if len(b05) > 0:
        archivos.append(guardar(b05, "B05_kruskal_irc_ccaa.csv",
                                 "Kruskal-Wallis IRC vs CCAA"))
    if len(b05_post) > 0:
        archivos.append(guardar(b05_post, "B05b_posthoc_ccaa.csv",
                                 "Post-hoc Bonferroni CCAA"))

    b07 = tabla_B07(irc)
    if len(b07) > 0:
        archivos.append(guardar(b07, "B07_concentracion_organo_ccaa.csv",
                                 "HHI concentración por órgano × CCAA"))

    b08 = tabla_B08(irc, raw)
    if len(b08) > 0:
        archivos.append(guardar(b08, "B08_irc_vs_oferta_unica_por_ccaa.csv",
                                 "IRC vs tasa oferta única por CCAA"))

    # Quality report
    qpath = REPORTS_DIR / "bivariado_quality.txt"
    with open(qpath, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("INFORME DE CALIDAD — ANÁLISIS BIVARIADO (Script 06)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Nivel de significación: α = {ALPHA}\n")
        f.write(f"Tablas generadas: {len(archivos)}\n\n")
        for p in archivos:
            f.write(f"  {p.name}\n")
        if len(b05) > 0:
            f.write("\n--- KW IRC vs CCAA ---\n")
            f.write(b05.to_string(index=False))

    log.info("Quality report: %s", qpath)
    log.info("=" * 65)
    log.info("Script 06 completado. SIGUIENTE: script_07_modelos_regresion.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
