"""
script_03_calcular_irc.py
=========================
Construye el Índice de Riesgo Compuesto (IRC) combinando los outputs
de los Scripts 01 y 02.

Fórmula base
------------
  IRC = wC * SC_norm + wB * SB_norm

  SC_norm = score_corporativo / 5    (rango [0,1])
  SB_norm = score_comportamiento / 6 (rango [0,1])
  wC = 0.4  |  wB = 0.6  (base; ver especificaciones alternativas)

Reglas de imputación
--------------------
  - Si SC no disponible (sin cobertura BORME):
      IRC = SB_norm  (wB = 1.0), documentado en irc_cobertura_borme
  - Si algún indicador de SB no calculable:
      se imputa a 0 en SB, documentado en irc_indicadores_faltantes
  - Si SB tampoco disponible: IRC = NaN (registro excluido del análisis)

Especificaciones de robustez (para apartado 4.e)
-------------------------------------------------
  irc_base     wC=0.4  wB=0.6
  irc_iguales  wC=0.5  wB=0.5
  irc_comp     wC=0.0  wB=1.0   (solo comportamiento)
  irc_corp     wC=1.0  wB=0.0   (solo corporativo)

Unidad de análisis
------------------
  (adjudicatario_key, ccaa, cpv2d)

Outputs
-------
  curated/irc_por_nif_ccaa_cpv.parquet         ← tabla principal
  curated/irc_agregado_territorial.parquet     ← mediana ponderada por CCAA
  curated/irc_agregado_sectorial.parquet       ← mediana ponderada por CPV2d
  reports/irc_quality.txt
  logs/script_03.log

Prerequisitos
-------------
  Script 01: Nacional/BORME/curated/flags_corporativos.parquet
  Script 02: Nacional/curated/indicadores_comportamiento_flags.parquet
"""

from __future__ import annotations
from pathlib import Path
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# ── RUTAS ─────────────────────────────────────────────────────────────────────
BASE = Path("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos")

FLAGS_CORP  = BASE / "Nacional/BORME/curated/flags_corporativos.parquet"
FLAGS_COMP  = BASE / "Nacional/curated/indicadores_comportamiento_flags.parquet"

OUT_DIR     = BASE / "Nacional/curated"
REPORTS_DIR = BASE / "Nacional/reports"
LOGS_DIR    = BASE / "Nacional/logs"
for d in [OUT_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_path = LOGS_DIR / "script_03.log"
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
# Pesos base
WC_BASE, WB_BASE = 0.4, 0.6

# Máximos teóricos para normalización
SC_MAX = 5.0   # flags corporativos F1-F5
SB_MAX = 6.0   # flags comportamiento B1-B6

# Número mínimo de contratos para incluir un grupo en las agregaciones
N_MIN_CONTRATOS = 3

# Top-N sectores CPV2d por importe para la agregación sectorial
TOP_N_SECTORES = 20


# ── PASO 1: Cargar flags corporativos ─────────────────────────────────────────
def cargar_flags_corporativos() -> pd.DataFrame:
    """
    Carga el output del Script 01.
    Columnas clave: adjudicatario_key (nif o nombre_norm),
    score_corporativo, borme_cobertura, flags_fiabilidad.
    """
    if not FLAGS_CORP.exists():
        log.error("flags_corporativos.parquet no encontrado: %s", FLAGS_CORP)
        log.error("Ejecuta script_01 antes de continuar.")
        raise FileNotFoundError(FLAGS_CORP)

    df = pd.read_parquet(FLAGS_CORP)
    log.info("Flags corporativos cargados: %s filas", f"{len(df):,}")

    # La clave del Script 01 es adjudicatario_nif o adjudicatario_nombre_norm
    # Necesitamos unificarla en 'adjudicatario_key' para el join con Script 02
    if "adjudicatario_nif" in df.columns and "adjudicatario_nombre_norm" in df.columns:
        df["adjudicatario_key"] = (
            df["adjudicatario_nif"].fillna(df["adjudicatario_nombre_norm"])
        )
    elif "adjudicatario_nif" in df.columns:
        df["adjudicatario_key"] = df["adjudicatario_nif"]
    elif "adjudicatario_nombre_norm" in df.columns:
        df["adjudicatario_key"] = df["adjudicatario_nombre_norm"]
    else:
        raise ValueError("flags_corporativos.parquet no tiene columna de clave de adjudicatario.")

    # Seleccionar solo las columnas necesarias para el join
    cols = ["adjudicatario_key", "score_corporativo",
            "borme_cobertura", "flags_fiabilidad",
            "F1","F2","F3","F4","F5"]
    cols_presentes = [c for c in cols if c in df.columns]
    return df[cols_presentes].copy()


# ── PASO 2: Cargar flags de comportamiento ────────────────────────────────────
def cargar_flags_comportamiento() -> pd.DataFrame:
    """
    Carga el output del Script 02.
    Columnas clave: adjudicatario_key, ccaa, cpv2d,
    score_comportamiento, flag_B1..flag_B6,
    n_contratos, importe_total.
    """
    if not FLAGS_COMP.exists():
        log.error("indicadores_comportamiento_flags.parquet no encontrado: %s",
                  FLAGS_COMP)
        log.error("Ejecuta script_02 antes de continuar.")
        raise FileNotFoundError(FLAGS_COMP)

    df = pd.read_parquet(FLAGS_COMP)
    log.info("Flags comportamiento cargados: %s filas", f"{len(df):,}")
    log.info("  Columnas: %s", list(df.columns))
    return df


# ── PASO 3: Join y construcción de la tabla base IRC ─────────────────────────
def construir_tabla_base(corp: pd.DataFrame,
                          comp: pd.DataFrame) -> pd.DataFrame:
    """
    Une flags corporativos (nivel adjudicatario)
    con flags de comportamiento (nivel adjudicatario × ccaa × cpv2d).

    El join es left desde comportamiento: todos los grupos
    (adj, ccaa, cpv2d) del Script 02 se conservan.
    Los corporativos se añaden donde hay match por adjudicatario_key.
    """
    log.info("Construyendo tabla base IRC...")

    # Join: comportamiento ← corporativos (many-to-one por adjudicatario_key)
    base = comp.merge(
        corp,
        on="adjudicatario_key",
        how="left",
        suffixes=("", "_corp"),
    )

    n_con_corp = base["score_corporativo"].notna().sum()
    n_total    = len(base)
    log.info("  Grupos totales (adj, ccaa, cpv2d): %s", f"{n_total:,}")
    log.info("  Con datos corporativos BORME:      %s (%.1f%%)",
             f"{n_con_corp:,}", 100 * n_con_corp / max(n_total, 1))
    log.info("  Sin datos corporativos:            %s (%.1f%%)",
             f"{n_total - n_con_corp:,}",
             100 * (n_total - n_con_corp) / max(n_total, 1))

    return base


# ── PASO 4: Calcular IRC ──────────────────────────────────────────────────────
def calcular_irc(base: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada fila (adj, ccaa, cpv2d):
    1. Normaliza SC y SB a [0,1]
    2. Aplica las 4 especificaciones de pesos
    3. Genera columnas de trazabilidad
    """
    df = base.copy()

    # ── Normalización ─────────────────────────────────────────────────────────
    df["SC_norm"] = pd.to_numeric(
        df["score_corporativo"], errors="coerce"
    ) / SC_MAX
    df["SB_norm"] = pd.to_numeric(
        df["score_comportamiento"], errors="coerce"
    ) / SB_MAX

    # Clamp a [0, 1] por seguridad
    df["SC_norm"] = df["SC_norm"].clip(0, 1)
    df["SB_norm"] = df["SB_norm"].clip(0, 1)

    # ── Columnas de trazabilidad ──────────────────────────────────────────────
    # ¿Hay cobertura BORME?
    df["irc_cobertura_borme"] = (
        df["borme_cobertura"].fillna(False).astype(bool)
        if "borme_cobertura" in df.columns
        else False
    )

    # ¿Qué indicadores de comportamiento estaban disponibles?
    flag_cols_b = [c for c in df.columns if c.startswith("flag_B")]
    df["irc_indicadores_faltantes"] = df[flag_cols_b].isna().sum(axis=1)

    # ── Función IRC genérica ──────────────────────────────────────────────────
    def irc_con_pesos(sc_norm: pd.Series, sb_norm: pd.Series,
                      wc: float, wb: float) -> pd.Series:
        """
        Calcula IRC respetando reglas de imputación:
        - Si SC nulo y SB disponible: wB = 1.0 (solo comportamiento)
        - Si SC disponible y SB nulo: wC = 1.0 (solo corporativo)
        - Si ambos nulos: NaN
        """
        sc_ok = sc_norm.notna()
        sb_ok = sb_norm.notna()

        irc = pd.Series(np.nan, index=sc_norm.index)

        # Ambos disponibles
        mask_ambos = sc_ok & sb_ok
        irc[mask_ambos] = (
            wc * sc_norm[mask_ambos] + wb * sb_norm[mask_ambos]
        )

        # Solo comportamiento
        mask_solo_b = ~sc_ok & sb_ok
        irc[mask_solo_b] = sb_norm[mask_solo_b]

        # Solo corporativo
        mask_solo_c = sc_ok & ~sb_ok
        irc[mask_solo_c] = sc_norm[mask_solo_c]

        return irc

    # ── 4 Especificaciones ────────────────────────────────────────────────────
    df["irc_base"]    = irc_con_pesos(df["SC_norm"], df["SB_norm"],
                                       WC_BASE, WB_BASE)
    df["irc_iguales"] = irc_con_pesos(df["SC_norm"], df["SB_norm"],
                                       0.5, 0.5)
    df["irc_comp"]    = irc_con_pesos(df["SC_norm"], df["SB_norm"],
                                       0.0, 1.0)
    df["irc_corp"]    = irc_con_pesos(df["SC_norm"], df["SB_norm"],
                                       1.0, 0.0)

    # ── Cuartil del IRC base dentro de cada grupo de referencia ───────────────
    # (ccaa, cpv2d) — usado para definir irc_alto en el modelo logit (Script 07)
    df["irc_cuartil"] = df.groupby(["ccaa","cpv2d"])["irc_base"].transform(
        lambda x: pd.qcut(x.rank(method="first"), q=4,
                           labels=[1,2,3,4], duplicates="drop")
        if x.notna().sum() >= 4 else pd.Series(np.nan, index=x.index)
    )
    df["irc_alto"] = (df["irc_cuartil"] == 4).astype(float)
    df.loc[df["irc_cuartil"].isna(), "irc_alto"] = np.nan

    log.info("IRC calculado. Distribución irc_base:")
    desc = df["irc_base"].describe()
    log.info("  n=%s  media=%.4f  p25=%.4f  p50=%.4f  p75=%.4f  max=%.4f",
             f"{int(desc['count']):,}",
             desc["mean"], desc["25%"], desc["50%"], desc["75%"], desc["max"])

    return df


# ── PASO 5: Agregación territorial ───────────────────────────────────────────
def agregar_territorial(irc: pd.DataFrame) -> pd.DataFrame:
    """
    Por CCAA: mediana del irc_base ponderada por importe_total.
    Solo grupos con n_contratos >= N_MIN_CONTRATOS.
    """
    log.info("Agregando por territorio...")

    sub = irc[
        irc["irc_base"].notna() &
        (irc["n_contratos"] >= N_MIN_CONTRATOS)
    ].copy()

    def mediana_ponderada(g):
        """Mediana ponderada: ordena por irc, acumula pesos, busca el 0.5."""
        g = g.dropna(subset=["irc_base","importe_total"])
        g = g[g["importe_total"] > 0]
        if len(g) == 0:
            return np.nan
        g = g.sort_values("irc_base")
        w = g["importe_total"].values
        cumw = np.cumsum(w)
        # Mediana ponderada: punto donde la suma acumulada cruza la mitad del total
        return g["irc_base"].iloc[np.searchsorted(cumw, cumw[-1] / 2)]

    terr = (
        sub.groupby("ccaa")
        .apply(lambda g: pd.Series({
            "irc_mediana_pond":      mediana_ponderada(g),
            "irc_media":             g["irc_base"].mean(),
            "irc_p75":               g["irc_base"].quantile(0.75),
            "n_adjudicatarios":      g["adjudicatario_key"].nunique(),
            "n_grupos":              len(g),
            "importe_total_ccaa":    g["importe_total"].sum(),
            "pct_irc_alto":          g["irc_alto"].mean(),
            # Robustez: correlación con otras especificaciones
            "corr_base_iguales":     g["irc_base"].corr(g["irc_iguales"]),
            "corr_base_comp":        g["irc_base"].corr(g["irc_comp"]),
        }))
        .reset_index()
    )

    terr = terr.sort_values("irc_mediana_pond", ascending=False)
    log.info("  CCAA procesadas: %s", len(terr))
    return terr


# ── PASO 6: Agregación sectorial ─────────────────────────────────────────────
def agregar_sectorial(irc: pd.DataFrame) -> pd.DataFrame:
    """
    Por CPV2d (top-N por importe): mediana del irc_base ponderada por importe.
    """
    log.info("Agregando por sector (CPV2d)...")

    sub = irc[
        irc["irc_base"].notna() &
        (irc["n_contratos"] >= N_MIN_CONTRATOS)
    ].copy()

    # Top N sectores por importe total
    top_cpv = (
        sub.groupby("cpv2d")["importe_total"]
        .sum()
        .nlargest(TOP_N_SECTORES)
        .index.tolist()
    )
    sub = sub[sub["cpv2d"].isin(top_cpv)].copy()

    def mediana_ponderada(g):
        g = g.dropna(subset=["irc_base","importe_total"])
        g = g[g["importe_total"] > 0]
        if len(g) == 0:
            return np.nan
        g = g.sort_values("irc_base")
        w = g["importe_total"].values
        cumw = np.cumsum(w)
        return g["irc_base"].iloc[np.searchsorted(cumw, cumw[-1] / 2)]

    sect = (
        sub.groupby("cpv2d")
        .apply(lambda g: pd.Series({
            "irc_mediana_pond":   mediana_ponderada(g),
            "irc_media":          g["irc_base"].mean(),
            "irc_p75":            g["irc_base"].quantile(0.75),
            "n_adjudicatarios":   g["adjudicatario_key"].nunique(),
            "n_grupos":           len(g),
            "importe_total_cpv":  g["importe_total"].sum(),
            "pct_irc_alto":       g["irc_alto"].mean(),
            "corr_base_comp":     g["irc_base"].corr(g["irc_comp"]),
        }))
        .reset_index()
    )

    sect = sect.sort_values("irc_mediana_pond", ascending=False)
    log.info("  Sectores CPV2d procesados: %s (top %s por importe)",
             len(sect), TOP_N_SECTORES)
    return sect


# ── PASO 7: Análisis de robustez rápido ───────────────────────────────────────
def analisis_robustez(irc: pd.DataFrame) -> dict:
    """
    Calcula correlaciones entre las 4 especificaciones del IRC
    a nivel global y por CCAA, para el apartado 4.e (validaciones).
    """
    log.info("Analizando robustez entre especificaciones...")

    sub = irc[["irc_base","irc_iguales","irc_comp","irc_corp"]].dropna()
    corrs = sub.corr(method="spearman")

    # Estabilidad territorial: correlación entre rankings CCAA
    terr_specs = {}
    for spec in ["irc_base","irc_iguales","irc_comp","irc_corp"]:
        terr_specs[spec] = (
            irc[irc[spec].notna() & (irc["n_contratos"] >= N_MIN_CONTRATOS)]
            .groupby("ccaa")[spec]
            .median()
        )
    terr_df = pd.DataFrame(terr_specs)
    terr_corr = terr_df.corr(method="spearman")

    return {
        "corr_global":      corrs,
        "corr_ranking_ccaa": terr_corr,
        "n_validos_global": len(sub),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("SCRIPT 03 — Índice de Riesgo Compuesto (IRC)")
    log.info("Inicio: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    # 1. Cargar inputs
    corp = cargar_flags_corporativos()
    comp = cargar_flags_comportamiento()

    # 2. Construir tabla base
    base = construir_tabla_base(corp, comp)

    # 3. Calcular IRC
    irc = calcular_irc(base)

    # 4. Agregar
    territorial = agregar_territorial(irc)
    sectorial   = agregar_sectorial(irc)

    # 5. Robustez
    robustez = analisis_robustez(irc)

    # 6. Guardar outputs
    irc_path  = OUT_DIR / "irc_por_nif_ccaa_cpv.parquet"
    terr_path = OUT_DIR / "irc_agregado_territorial.parquet"
    sect_path = OUT_DIR / "irc_agregado_sectorial.parquet"

    irc.to_parquet(irc_path,   index=False)
    territorial.to_parquet(terr_path, index=False)
    sectorial.to_parquet(sect_path,   index=False)

    log.info("Guardado IRC:         %s  (%s filas)", irc_path,  f"{len(irc):,}")
    log.info("Guardado territorial: %s  (%s CCAA)",  terr_path, len(territorial))
    log.info("Guardado sectorial:   %s  (%s CPV2d)", sect_path, len(sectorial))

    # 7. Quality report
    n = len(irc)
    n_validos = int(irc["irc_base"].notna().sum())

    qpath = REPORTS_DIR / "irc_quality.txt"
    with open(qpath, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("INFORME DE CALIDAD — IRC (Script 03)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 65 + "\n\n")

        f.write("--- Cobertura ---\n")
        f.write(f"  Grupos (adj, ccaa, cpv2d) totales:  {n:,}\n")
        f.write(f"  Con IRC base calculado:             {n_validos:,}  "
                f"({100*n_validos/max(n,1):.1f}%)\n")
        f.write(f"  Con cobertura BORME (SC disponible): "
                f"{int(irc['irc_cobertura_borme'].sum()):,}\n\n")

        f.write("--- Distribución IRC base ---\n")
        desc = irc["irc_base"].describe()
        for k in ["count","mean","std","min","25%","50%","75%","max"]:
            f.write(f"  {k:>6}: {desc[k]:.4f}\n")

        f.write("\n--- Distribución por cuartil (irc_cuartil) ---\n")
        for q, cnt in irc["irc_cuartil"].value_counts().sort_index().items():
            f.write(f"  Q{q}: {int(cnt):,}  ({100*cnt/n:.1f}%)\n")

        f.write("\n--- Especificaciones alternativas (media global) ---\n")
        for spec in ["irc_base","irc_iguales","irc_comp","irc_corp"]:
            if spec in irc.columns:
                f.write(f"  {spec}: media={irc[spec].mean():.4f}  "
                        f"p50={irc[spec].median():.4f}\n")

        f.write("\n--- Robustez: correlación Spearman entre especificaciones ---\n")
        f.write(robustez["corr_global"].round(4).to_string())

        f.write("\n\n--- Robustez: correlación rankings territoriales (CCAA) ---\n")
        f.write(robustez["corr_ranking_ccaa"].round(4).to_string())

        f.write("\n\n--- Top 10 CCAA por IRC mediana ponderada ---\n")
        top_terr = territorial.head(10)[
            ["ccaa","irc_mediana_pond","n_adjudicatarios","importe_total_ccaa"]
        ]
        f.write(top_terr.to_string(index=False))

        f.write("\n\n--- Top 10 sectores CPV2d por IRC mediana ponderada ---\n")
        top_sect = sectorial.head(10)[
            ["cpv2d","irc_mediana_pond","n_adjudicatarios","importe_total_cpv"]
        ]
        f.write(top_sect.to_string(index=False))

        f.write(f"\n\nOutputs:\n  {irc_path}\n  {terr_path}\n  {sect_path}\n")

    log.info("Quality report: %s", qpath)
    log.info("=" * 65)
    log.info("Script 03 completado.")
    log.info("SIGUIENTE: script_04_tablas_descriptivas.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
