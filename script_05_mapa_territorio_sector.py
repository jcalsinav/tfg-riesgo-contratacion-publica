"""
script_05_mapa_territorio_sector.py
====================================
Genera las tablas de análisis territorial y sectorial del IRC
para los apartados 5.b y 5.c del TFG.

Inputs
------
  Nacional/curated/irc_por_nif_ccaa_cpv.parquet       (Script 03)
  Nacional/curated/irc_agregado_territorial.parquet    (Script 03)
  Nacional/curated/irc_agregado_sectorial.parquet      (Script 03)

Outputs  (Nacional/outputs/mapas/)
-------
  M01_irc_por_ccaa_completo.csv         ← tabla completa para mapa coroplético
  M02_irc_por_cpv2d_top20.csv           ← tabla sectorial top 20
  M03_flags_activos_por_ccaa.csv        ← nº flags activos por CCAA y tipo
  M04_flags_activos_por_sector.csv      ← nº flags activos por CPV2d y tipo
  M05_irc_estabilidad_temporal.csv      ← IRC subperíodos 2019-21 vs 2022-25
  M06_concentracion_top5_por_ccaa.csv   ← top-5 share del gasto por CCAA
  mapas_quality.txt
  logs/script_05.log

Nota sobre anonimato
--------------------
Ninguna tabla incluye identificadores de empresas ni nombres.
Todas las métricas son agregadas por (CCAA, sector) o (sector).
"""

from __future__ import annotations
from pathlib import Path
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# ── RUTAS ─────────────────────────────────────────────────────────────────────
BASE = Path("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos")

IRC_PATH  = BASE / "Nacional/curated/irc_por_nif_ccaa_cpv.parquet"
RAW_PATH  = BASE / "Nacional/curated/indicadores_comportamiento_raw.parquet"
TERR_PATH = BASE / "Nacional/curated/irc_agregado_territorial.parquet"
SECT_PATH = BASE / "Nacional/curated/irc_agregado_sectorial.parquet"

OUT_DIR     = BASE / "Nacional/outputs/mapas"
REPORTS_DIR = BASE / "Nacional/reports"
LOGS_DIR    = BASE / "Nacional/logs"
for d in [OUT_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_path = LOGS_DIR / "script_05.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── CONSTANTES ────────────────────────────────────────────────────────────────
N_MIN = 3          # mínimo de contratos para incluir un grupo
TOP_N = 20         # top sectores CPV2d

# Nombres legibles de CCAA para las tablas del TFG
CCAA_NOMBRES = {
    "AN": "Andalucía", "CT": "Cataluña", "MD": "Comunidad de Madrid",
    "PV": "País Vasco", "GA": "Galicia",  "VC": "C. Valenciana",
    "NAC": "Nacional (PLACSP)",
}


# ── UTILIDADES ────────────────────────────────────────────────────────────────
def guardar(df: pd.DataFrame, nombre: str, desc: str) -> Path:
    ruta = OUT_DIR / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    log.info("  [OK] %s  (%s filas) — %s", nombre, f"{len(df):,}", desc)
    return ruta


def mediana_ponderada(valores: pd.Series, pesos: pd.Series) -> float:
    df = pd.DataFrame({"v": valores, "w": pesos}).dropna()
    df = df[df["w"] > 0].sort_values("v")
    if len(df) == 0:
        return np.nan
    cumw = df["w"].cumsum()
    return df["v"].iloc[np.searchsorted(cumw.values, cumw.values[-1] / 2)]


# ── M01: IRC completo por CCAA ────────────────────────────────────────────────
def tabla_M01(irc: pd.DataFrame) -> pd.DataFrame:
    """
    Una fila por CCAA con todas las métricas del IRC:
    mediana ponderada, media, p75, % con irc_alto,
    correlaciones entre especificaciones y nº adjudicatarios.
    """
    log.info("M01: IRC por CCAA completo...")

    sub = irc[
        irc["irc_base"].notna() &
        (irc.get("n_contratos", pd.Series(1, index=irc.index)) >= N_MIN)
    ].copy()

    flag_cols = [c for c in irc.columns if c.startswith("flag_")]

    rows = []
    for ccaa, g in sub.groupby("ccaa"):
        row = {
            "ccaa":              ccaa,
            "ccaa_nombre":       CCAA_NOMBRES.get(ccaa, ccaa),
            "n_grupos":          len(g),
            "n_adjudicatarios":  g["adjudicatario_key"].nunique(),
            "importe_total":     g["importe_total"].sum() if "importe_total" in g.columns else np.nan,
            "irc_mediana_pond":  mediana_ponderada(
                g["irc_base"],
                g["importe_total"] if "importe_total" in g.columns
                else pd.Series(1, index=g.index)
            ),
            "irc_media":         g["irc_base"].mean(),
            "irc_p25":           g["irc_base"].quantile(0.25),
            "irc_p75":           g["irc_base"].quantile(0.75),
            "pct_irc_alto":      g["irc_alto"].mean() * 100 if "irc_alto" in g.columns else np.nan,
            # Robustez: correlación base vs solo comportamiento
            "corr_base_comp":    g["irc_base"].corr(g["irc_comp"])
                                 if "irc_comp" in g.columns else np.nan,
            "corr_base_iguales": g["irc_base"].corr(g["irc_iguales"])
                                 if "irc_iguales" in g.columns else np.nan,
        }
        # Tasa de activación de cada flag
        for fc in flag_cols:
            if fc in g.columns:
                row[f"pct_{fc}"] = round(g[fc].mean() * 100, 2)
        rows.append(row)

    m01 = pd.DataFrame(rows).sort_values("irc_mediana_pond", ascending=False)
    return m01


# ── M02: IRC por sector CPV2d ─────────────────────────────────────────────────
def tabla_M02(irc: pd.DataFrame) -> pd.DataFrame:
    """
    Top-N sectores por importe con IRC y distribución de flags.
    """
    log.info("M02: IRC por sector CPV2d...")

    sub = irc[
        irc["irc_base"].notna() &
        (irc.get("n_contratos", pd.Series(1, index=irc.index)) >= N_MIN)
    ].copy()

    # Top sectores por importe
    top_cpv = (
        sub.groupby("cpv2d")["importe_total"].sum()
        .nlargest(TOP_N).index.tolist()
        if "importe_total" in sub.columns
        else sub["cpv2d"].value_counts().head(TOP_N).index.tolist()
    )
    sub = sub[sub["cpv2d"].isin(top_cpv)].copy()

    flag_cols = [c for c in irc.columns if c.startswith("flag_")]

    rows = []
    for cpv2d, g in sub.groupby("cpv2d"):
        row = {
            "cpv2d":             cpv2d,
            "n_grupos":          len(g),
            "n_adjudicatarios":  g["adjudicatario_key"].nunique(),
            "importe_total":     g["importe_total"].sum() if "importe_total" in g.columns else np.nan,
            "irc_mediana_pond":  mediana_ponderada(
                g["irc_base"],
                g["importe_total"] if "importe_total" in g.columns
                else pd.Series(1, index=g.index)
            ),
            "irc_media":         g["irc_base"].mean(),
            "irc_p75":           g["irc_base"].quantile(0.75),
            "pct_irc_alto":      g["irc_alto"].mean() * 100 if "irc_alto" in g.columns else np.nan,
        }
        for fc in flag_cols:
            if fc in g.columns:
                row[f"pct_{fc}"] = round(g[fc].mean() * 100, 2)
        rows.append(row)

    m02 = pd.DataFrame(rows).sort_values("irc_mediana_pond", ascending=False)
    return m02


# ── M03: Flags activos por CCAA ───────────────────────────────────────────────
def tabla_M03(irc: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada CCAA, porcentaje de adjudicatarios con cada flag activo.
    Formato largo: (ccaa, flag, pct_activos).
    """
    log.info("M03: flags activos por CCAA...")

    flag_cols = [c for c in irc.columns if c.startswith("flag_")]
    if not flag_cols:
        log.warning("  No hay columnas flag_* en el IRC.")
        return pd.DataFrame()

    rows = []
    for ccaa, g in irc.groupby("ccaa"):
        for fc in flag_cols:
            if fc not in g.columns:
                continue
            n_valido = g[fc].notna().sum()
            if n_valido == 0:
                continue
            rows.append({
                "ccaa":        ccaa,
                "ccaa_nombre": CCAA_NOMBRES.get(ccaa, ccaa),
                "flag":        fc,
                "n_validos":   int(n_valido),
                "n_activos":   int(g[fc].sum()),
                "pct_activos": round(100 * g[fc].mean(), 2),
            })

    return pd.DataFrame(rows).sort_values(["flag","pct_activos"], ascending=[True, False])


# ── M04: Flags activos por sector ─────────────────────────────────────────────
def tabla_M04(irc: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada CPV2d (top-N), porcentaje de adjudicatarios con cada flag activo.
    """
    log.info("M04: flags activos por sector...")

    flag_cols = [c for c in irc.columns if c.startswith("flag_")]
    if not flag_cols:
        return pd.DataFrame()

    top_cpv = (
        irc.groupby("cpv2d")["importe_total"].sum()
        .nlargest(TOP_N).index.tolist()
        if "importe_total" in irc.columns
        else irc["cpv2d"].value_counts().head(TOP_N).index.tolist()
    )
    sub = irc[irc["cpv2d"].isin(top_cpv)].copy()

    rows = []
    for cpv2d, g in sub.groupby("cpv2d"):
        for fc in flag_cols:
            if fc not in g.columns:
                continue
            n_valido = g[fc].notna().sum()
            if n_valido == 0:
                continue
            rows.append({
                "cpv2d":       cpv2d,
                "flag":        fc,
                "n_validos":   int(n_valido),
                "n_activos":   int(g[fc].sum()),
                "pct_activos": round(100 * g[fc].mean(), 2),
            })

    return pd.DataFrame(rows).sort_values(["flag","pct_activos"], ascending=[True, False])


# ── M05: Estabilidad temporal ─────────────────────────────────────────────────
def tabla_M05(irc: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula IRC medio por subperíodo (2019-2021 vs 2022-2025) y compara
    rankings territoriales mediante correlación de Spearman.

    Metodología:
    - Las fechas (fecha_min) no están en el IRC sino en el raw de comportamiento.
      Se hace un join por (adjudicatario_key, ccaa, cpv2d) para incorporarlas.
    - Se usa la MEDIA del IRC (no la mediana) porque la distribución del IRC
      es muy sesgada: la mediana es 0 en casi todas las CCAA, lo que colapsa
      el ranking. La media captura la varianza entre CCAA necesaria para
      evaluar estabilidad.
    - Período asignado según fecha_min del grupo (primera adjudicación).
    - Criterio de robustez: ρ Spearman ≥ 0.70.
    """
    log.info("M05: estabilidad temporal 2019-21 vs 2022-25...")

    # Cargar fechas desde el raw de comportamiento
    if not RAW_PATH.exists():
        log.warning("  M05: raw de comportamiento no encontrado (%s). Se omite.", RAW_PATH)
        return pd.DataFrame()

    log.info("  Cargando fechas desde raw de comportamiento...")
    raw_fechas = pd.read_parquet(
        RAW_PATH,
        columns=["adjudicatario_key", "ccaa", "cpv2d", "fecha_min"]
    )

    # Join IRC ← fechas (left join, 1:1 garantizado)
    irc_f = irc.merge(raw_fechas, on=["adjudicatario_key", "ccaa", "cpv2d"], how="left")
    irc_f["fecha_min"] = pd.to_datetime(irc_f["fecha_min"], errors="coerce", utc=True)
    irc_f["_anio_min"] = irc_f["fecha_min"].dt.year

    n_con_fecha = irc_f["_anio_min"].notna().sum()
    log.info("  Grupos con fecha_min: %s / %s (%.1f%%)",
             f"{n_con_fecha:,}", f"{len(irc_f):,}", 100 * n_con_fecha / len(irc_f))

    # Asignar período según fecha_min del grupo
    irc_f["_periodo"] = pd.cut(
        irc_f["_anio_min"],
        bins=[2018, 2021, 2025],
        labels=["2019-2021", "2022-2025"],
    )

    sub = irc_f[irc_f["irc_base"].notna() & irc_f["_periodo"].notna()].copy()
    log.info("  Grupos con período asignado: %s (P1=%s, P2=%s)",
             f"{len(sub):,}",
             f"{(sub['_periodo']=='2019-2021').sum():,}",
             f"{(sub['_periodo']=='2022-2025').sum():,}")

    # Media IRC por CCAA × período
    # (la mediana colapsa a 0/0.08 por distribución bimodal — sin varianza útil)
    terr = (
        sub.groupby(["ccaa", "_periodo"])["irc_base"]
        .mean()
        .reset_index()
        .rename(columns={"irc_base": "irc_media"})
    )

    # Tabla pivotada para comparar directamente
    pivot = terr.pivot(index="ccaa", columns="_periodo", values="irc_media")
    pivot.columns = ["irc_media_2019_2021", "irc_media_2022_2025"]
    pivot = pivot.reset_index()
    pivot["ccaa_nombre"] = pivot["ccaa"].map(CCAA_NOMBRES).fillna(pivot["ccaa"])
    pivot["delta_irc"] = (pivot["irc_media_2022_2025"] - pivot["irc_media_2019_2021"]).round(4)

    # Correlación de Spearman entre rankings
    col_p1, col_p2 = "irc_media_2019_2021", "irc_media_2022_2025"
    n_validos = pivot[[col_p1, col_p2]].dropna().shape[0]
    if n_validos >= 3:
        corr = pivot[col_p1].corr(pivot[col_p2], method="spearman")
        pivot["corr_spearman_rankings"] = round(corr, 4)
        robustez = "ROBUSTO (ρ ≥ 0.70)" if corr >= 0.70 else f"NO CUMPLE (umbral 0.70)"
        log.info("  Correlación Spearman rankings territoriales: ρ=%.4f → %s", corr, robustez)
    else:
        log.warning("  M05: insuficientes CCAA con datos en ambos períodos (%d).", n_validos)

    return pivot.sort_values(col_p2, ascending=False, na_position="last")


# ── M06: Concentración top-5 por CCAA ────────────────────────────────────────
def tabla_M06(irc: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada CCAA: cuánto gasto acumulan los 5 adjudicatarios
    con más importe (top-5 share). Métrica de concentración de mercado.
    """
    log.info("M06: concentración top-5 por CCAA...")

    if "importe_total" not in irc.columns:
        log.warning("  M06: importe_total no disponible.")
        return pd.DataFrame()

    sub = irc[irc["importe_total"].notna() & (irc["importe_total"] > 0)].copy()

    rows = []
    for ccaa, g in sub.groupby("ccaa"):
        gasto_total = g["importe_total"].sum()
        if gasto_total == 0:
            continue
        gasto_top5 = (
            g.groupby("adjudicatario_key")["importe_total"]
            .sum()
            .nlargest(5)
            .sum()
        )
        gasto_top1 = (
            g.groupby("adjudicatario_key")["importe_total"]
            .sum()
            .nlargest(1)
            .sum()
        )
        rows.append({
            "ccaa":             ccaa,
            "ccaa_nombre":      CCAA_NOMBRES.get(ccaa, ccaa),
            "gasto_total":      round(gasto_total, 0),
            "top1_share_pct":   round(100 * gasto_top1 / gasto_total, 2),
            "top5_share_pct":   round(100 * gasto_top5 / gasto_total, 2),
            "n_adjudicatarios": g["adjudicatario_key"].nunique(),
        })

    return pd.DataFrame(rows).sort_values("top5_share_pct", ascending=False)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("SCRIPT 05 — Mapa territorial y sectorial")
    log.info("Inicio: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    if not IRC_PATH.exists():
        log.error("irc_por_nif_ccaa_cpv.parquet no encontrado. Ejecuta script_03 primero.")
        raise FileNotFoundError(IRC_PATH)

    irc = pd.read_parquet(IRC_PATH)
    log.info("IRC cargado: %s filas", f"{len(irc):,}")

    archivos = []

    m01 = tabla_M01(irc)
    archivos.append(guardar(m01, "M01_irc_por_ccaa_completo.csv",
                             "IRC completo por CCAA"))

    m02 = tabla_M02(irc)
    archivos.append(guardar(m02, "M02_irc_por_cpv2d_top20.csv",
                             "IRC por sector CPV2d (top 20)"))

    m03 = tabla_M03(irc)
    if len(m03) > 0:
        archivos.append(guardar(m03, "M03_flags_activos_por_ccaa.csv",
                                 "Flags activos por CCAA"))

    m04 = tabla_M04(irc)
    if len(m04) > 0:
        archivos.append(guardar(m04, "M04_flags_activos_por_sector.csv",
                                 "Flags activos por sector CPV2d"))

    m05 = tabla_M05(irc)
    if len(m05) > 0:
        archivos.append(guardar(m05, "M05_irc_estabilidad_temporal.csv",
                                 "Estabilidad temporal 2019-21 vs 2022-25"))

    m06 = tabla_M06(irc)
    if len(m06) > 0:
        archivos.append(guardar(m06, "M06_concentracion_top5_por_ccaa.csv",
                                 "Concentración top-5 share por CCAA"))

    qpath = REPORTS_DIR / "mapas_quality.txt"
    with open(qpath, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("INFORME DE CALIDAD — MAPAS (Script 05)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Tablas generadas: {len(archivos)}\n")
        for p in archivos:
            f.write(f"  {p.name}\n")
        f.write("\n--- Resumen M01 (CCAA × IRC mediana ponderada) ---\n")
        cols_res = ["ccaa_nombre","irc_mediana_pond","n_adjudicatarios","pct_irc_alto"]
        cols_ok  = [c for c in cols_res if c in m01.columns]
        f.write(m01[cols_ok].to_string(index=False))

    if len(m05) > 0 and "corr_spearman_rankings" in m05.columns:
        corr_val = m05["corr_spearman_rankings"].iloc[0]
        robustez = "ROBUSTO (ρ ≥ 0.70)" if corr_val >= 0.70 else "NO CUMPLE (umbral 0.70)"
        with open(qpath, "a", encoding="utf-8") as f:
            f.write("\n\n--- Resumen M05 (Estabilidad temporal) ---\n")
            f.write(f"  Correlación Spearman rankings 2019-21 vs 2022-25: ρ={corr_val:.4f}\n")
            f.write(f"  Criterio robustez (ρ ≥ 0.70): {robustez}\n")
            f.write("  Nota: IRC calculado como media por CCAA × periodo.\n")
            f.write("  La mediana es inviable (colapsa a 0 por distribución bimodal del IRC).\n")

    log.info("Quality report: %s", qpath)
    log.info("=" * 65)
    log.info("Script 05 completado. SIGUIENTE: script_06_analisis_bivariado.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
