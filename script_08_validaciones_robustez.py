"""
script_08_validaciones_robustez.py
====================================
Ejecuta todas las pruebas de robustez del apartado 4.e del TFG.

Pruebas
-------
1. Sensibilidad de pesos del IRC (wC/wB)
   → Correlación Spearman entre rankings CCAA y sectores bajo 4 especificaciones
2. Sensibilidad del umbral de alerta (p60 / p75 / p90)
   → Recalcula flags con distintos percentiles y compara top-CCAA
3. Estabilidad temporal (2019-2021 vs 2022-2025)
   → Correlación entre rankings en ambos subperíodos
4. Coherencia con TED
   → Adjudicatarios con flag_B5 alto: ¿tienen menor tasa de publicación en TED?

Inputs
------
  Nacional/curated/irc_por_nif_ccaa_cpv.parquet
  Nacional/curated/indicadores_comportamiento_raw.parquet
  Nacional/curated/irc_agregado_territorial.parquet
  (Opcional) ted/ted_es_can.parquet + ted/crossval_sara_v2.parquet

Outputs  (Nacional/outputs/robustez/)
-------
  V01_sensibilidad_pesos_rankings_ccaa.csv
  V02_sensibilidad_pesos_rankings_sector.csv
  V03_sensibilidad_umbral_flags.csv
  V04_estabilidad_temporal_ccaa.csv
  V05_coherencia_ted_sara.csv              ← solo si TED disponible
  robustez_quality.txt
  logs/script_08.log
"""

from __future__ import annotations
from pathlib import Path
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos")

IRC_PATH  = BASE / "Nacional/curated/irc_por_nif_ccaa_cpv.parquet"
RAW_PATH  = BASE / "Nacional/curated/indicadores_comportamiento_raw.parquet"
TERR_PATH = BASE / "Nacional/curated/irc_agregado_territorial.parquet"

# TED (opcionales)
TED_PATH     = BASE / "Nacional/TED/raw/ted_es_can.parquet"
SARA_PATH    = BASE / "Nacional/TED/raw/crossval_sara_v2.parquet"

OUT_DIR     = BASE / "Nacional/outputs/robustez"
REPORTS_DIR = BASE / "Nacional/reports"
LOGS_DIR    = BASE / "Nacional/logs"
for d in [OUT_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

log_path = LOGS_DIR / "script_08.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── UTILIDADES ────────────────────────────────────────────────────────────────
def guardar(df: pd.DataFrame, nombre: str, desc: str) -> Path:
    ruta = OUT_DIR / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    log.info("  [OK] %s  (%s filas) — %s", nombre, f"{len(df):,}", desc)
    return ruta


def spearman_ranking(s1: pd.Series, s2: pd.Series) -> float:
    df = pd.DataFrame({"a": s1, "b": s2}).dropna()
    if len(df) < 3:
        return np.nan
    rho, _ = stats.spearmanr(df["a"], df["b"])
    return round(rho, 4)


# ── V01/V02: Sensibilidad de pesos ────────────────────────────────────────────
def sensibilidad_pesos(irc: pd.DataFrame):
    """
    Compara rankings territoriales y sectoriales entre las 4 especificaciones.
    Si los rankings son estables, los resultados son robustos a la elección de pesos.
    """
    log.info("V01/V02: sensibilidad de pesos...")

    specs = ["irc_base","irc_iguales","irc_comp","irc_corp"]
    specs_ok = [s for s in specs if s in irc.columns]

    # Rankings por CCAA (mediana por spec)
    terr_specs = {}
    for sp in specs_ok:
        terr_specs[sp] = (
            irc[irc[sp].notna()]
            .groupby("ccaa")[sp]
            .median()
        )
    terr_df = pd.DataFrame(terr_specs)

    # Matriz de correlaciones entre rankings
    corr_rows = []
    for i, sp1 in enumerate(specs_ok):
        for sp2 in specs_ok[i:]:
            rho = spearman_ranking(terr_df.get(sp1), terr_df.get(sp2))
            corr_rows.append({
                "esp_A": sp1,
                "esp_B": sp2,
                "corr_ranking_ccaa": rho,
                "interpretacion": "muy alta" if abs(rho) >= 0.9
                                  else "alta" if abs(rho) >= 0.7
                                  else "moderada" if abs(rho) >= 0.5
                                  else "baja",
            })
    v01 = pd.DataFrame(corr_rows)

    # Rankings por sector (CPV2d)
    sect_specs = {}
    for sp in specs_ok:
        sect_specs[sp] = (
            irc[irc[sp].notna()]
            .groupby("cpv2d")[sp]
            .median()
        )
    sect_df = pd.DataFrame(sect_specs)

    corr_rows_s = []
    for i, sp1 in enumerate(specs_ok):
        for sp2 in specs_ok[i:]:
            rho = spearman_ranking(sect_df.get(sp1), sect_df.get(sp2))
            corr_rows_s.append({
                "esp_A": sp1,
                "esp_B": sp2,
                "corr_ranking_sector": rho,
            })
    v02 = pd.DataFrame(corr_rows_s)

    return v01, v02


# ── V03: Sensibilidad del umbral de activación ───────────────────────────────
def sensibilidad_umbral(irc: pd.DataFrame, raw: pd.DataFrame):
    """
    Recalcula flags con percentiles 60, 75 (base) y 90.
    Compara qué CCAA quedan en el top por cada umbral.
    """
    log.info("V03: sensibilidad umbral de activación...")

    flag_raw_cols = {
        "flag_B1": "B1_ratio",
        "flag_B2": "B2_hhi_medio",
        "flag_B3": "B3_racha_max",
        "flag_B6": "B6_ratio",
    }

    resultados = []
    percentiles = [0.60, 0.75, 0.90]

    for pct in percentiles:
        score_list = []
        for flag_col, raw_col in flag_raw_cols.items():
            if raw_col not in raw.columns:
                continue
            merged = irc[["adjudicatario_key","ccaa","cpv2d"]].merge(
                raw[["adjudicatario_key","ccaa","cpv2d",raw_col]],
                on=["adjudicatario_key","ccaa","cpv2d"],
                how="left",
            )
            p_val = merged.groupby(["ccaa","cpv2d"])[raw_col].transform(
                lambda x: x.quantile(pct)
            )
            flag = (merged[raw_col] > p_val).astype(float)
            flag[merged[raw_col].isna()] = np.nan
            score_list.append(flag)

        if not score_list:
            continue

        score_df = pd.concat(score_list, axis=1)
        score_df.columns = [f"f_{i}" for i in range(len(score_list))]
        score_combinado = score_df.sum(axis=1, min_count=1)

        # Mediana del score por CCAA
        ccaa_irc = irc["ccaa"].copy()
        por_ccaa = (
            pd.DataFrame({"ccaa": ccaa_irc, "score": score_combinado})
            .groupby("ccaa")["score"]
            .median()
            .reset_index()
            .rename(columns={"score": f"score_p{int(pct*100)}"})
        )
        resultados.append(por_ccaa.set_index("ccaa"))

    if not resultados:
        return pd.DataFrame()

    v03 = pd.concat(resultados, axis=1).reset_index()

    # Correlaciones entre umbrales
    cols_score = [c for c in v03.columns if c.startswith("score_")]
    if len(cols_score) >= 2:
        for i in range(len(cols_score)):
            for j in range(i+1, len(cols_score)):
                rho = spearman_ranking(v03[cols_score[i]], v03[cols_score[j]])
                log.info("  Corr ranking %s vs %s: %.4f",
                         cols_score[i], cols_score[j], rho)

    return v03


# ── V04: Estabilidad temporal ─────────────────────────────────────────────────
def estabilidad_temporal(irc: pd.DataFrame):
    """
    Calcula IRC por subperíodo y correlación de rankings.
    Las fechas (fecha_min) no están en el IRC sino en el raw de comportamiento.
    Se hace un join por (adjudicatario_key, ccaa, cpv2d) para incorporarlas.
    """
    log.info("V04: estabilidad temporal...")

    if not RAW_PATH.exists():
        log.warning("  V04: raw de comportamiento no encontrado (%s). Se omite.", RAW_PATH)
        return pd.DataFrame()

    log.info("  Cargando fechas desde raw de comportamiento...")
    raw_fechas = pd.read_parquet(
        RAW_PATH,
        columns=["adjudicatario_key", "ccaa", "cpv2d", "fecha_min"]
    )
    irc_f = irc.merge(raw_fechas, on=["adjudicatario_key", "ccaa", "cpv2d"], how="left")
    irc_f["fecha_min"] = pd.to_datetime(irc_f["fecha_min"], errors="coerce", utc=True)
    irc_f["_anio_min"] = irc_f["fecha_min"].dt.year

    n_con_fecha = irc_f["_anio_min"].notna().sum()
    log.info("  Grupos con fecha_min: %s / %s (%.1f%%)",
             f"{n_con_fecha:,}", f"{len(irc_f):,}", 100 * n_con_fecha / len(irc_f))

    sub_a = irc_f[irc_f["_anio_min"].between(2019, 2021) & irc_f["irc_base"].notna()]
    sub_b = irc_f[irc_f["_anio_min"].between(2022, 2025) & irc_f["irc_base"].notna()]

    rank_a = sub_a.groupby("ccaa")["irc_base"].mean().rename("irc_2019_2021")
    rank_b = sub_b.groupby("ccaa")["irc_base"].mean().rename("irc_2022_2025")

    v04 = pd.concat([rank_a, rank_b], axis=1).reset_index()
    v04 = v04.rename(columns={"ccaa": "ccaa"})

    rho = spearman_ranking(v04["irc_2019_2021"], v04["irc_2022_2025"])
    v04["corr_spearman_subperiodos"] = rho
    log.info("  Correlación rankings territoriales 2019-21 vs 2022-25: %.4f", rho)

    return v04.sort_values("irc_2022_2025", ascending=False)


# ── V05: Coherencia con TED ───────────────────────────────────────────────────
def coherencia_ted(irc: pd.DataFrame) -> pd.DataFrame:
    """
    Para adjudicatarios con flag_B5 alto (clustering bajo umbral SARA):
    ¿tienen menor tasa de publicación en TED que el grupo de referencia?
    Solo se ejecuta si los ficheros TED están disponibles.
    """
    log.info("V05: coherencia con TED...")

    if not SARA_PATH.exists():
        log.warning("  V05: crossval_sara_v2.parquet no encontrado en %s", SARA_PATH)
        log.warning("  Buscando en rutas alternativas...")

        # Buscar en rutas alternativas comunes
        alt_paths = [
            BASE / "Nacional/TED/crossval_sara_v2.parquet",
            BASE / "Nacional/PLACSP/raw/crossval_sara_v2.parquet",
            BASE / "ted/crossval_sara_v2.parquet",
        ]
        sara_found = next((p for p in alt_paths if p.exists()), None)
        if sara_found is None:
            log.warning("  V05: TED cross-validation no disponible. Se omite.")
            return pd.DataFrame()
        sara_path = sara_found
    else:
        sara_path = SARA_PATH

    log.info("  Cargando TED cross-validation desde %s", sara_path)
    sara = pd.read_parquet(sara_path)
    log.info("  SARA: %s filas | Columnas: %s", f"{len(sara):,}", list(sara.columns)[:10])

    if "_ted_missing" not in sara.columns:
        log.warning("  V05: columna _ted_missing no encontrada en SARA.")
        return pd.DataFrame()

    # Adjudicatarios con flag_B5 activo
    if "flag_B5" not in irc.columns:
        log.warning("  V05: flag_B5 no disponible en IRC.")
        return pd.DataFrame()

    nifs_b5_alto  = set(irc[irc["flag_B5"] == 1.0]["adjudicatario_key"].dropna())
    nifs_b5_bajo  = set(irc[irc["flag_B5"] == 0.0]["adjudicatario_key"].dropna())

    # Columna NIF en SARA
    col_nif_sara = next(
        (c for c in ["nif_adjudicatario","adjudicatario_nif","nif"]
         if c in sara.columns), None
    )
    if col_nif_sara is None:
        log.warning("  V05: no hay columna NIF en SARA.")
        return pd.DataFrame()

    sara["_nif_norm"] = (
        sara[col_nif_sara].astype(str).str.upper()
        .str.replace(r"[\s\-\.]", "", regex=True)
    )
    sara["_grupo_b5"] = sara["_nif_norm"].map(
        lambda x: "alto" if x in nifs_b5_alto
                  else "bajo" if x in nifs_b5_bajo
                  else "sin_irc"
    )

    v05 = (
        sara.groupby("_grupo_b5")
        .agg(
            n_contratos_sara=("_ted_missing","count"),
            n_missing_ted   =("_ted_missing","sum"),
            tasa_missing_pct=(
                "_ted_missing",
                lambda x: round(100 * x.mean(), 2)
            ),
        )
        .reset_index()
        .rename(columns={"_grupo_b5": "grupo_flag_B5"})
    )

    log.info("  V05: tasa missing TED por grupo flag_B5:")
    log.info(v05.to_string(index=False))

    return v05


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("SCRIPT 08 — Validaciones y robustez")
    log.info("Inicio: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    if not IRC_PATH.exists():
        raise FileNotFoundError(f"IRC no encontrado: {IRC_PATH}")

    irc = pd.read_parquet(IRC_PATH)
    raw = pd.read_parquet(RAW_PATH) if RAW_PATH.exists() else pd.DataFrame()

    log.info("IRC: %s filas | Raw: %s filas",
             f"{len(irc):,}", f"{len(raw):,}")

    archivos = []

    v01, v02 = sensibilidad_pesos(irc)
    if len(v01) > 0:
        archivos.append(guardar(v01, "V01_sensibilidad_pesos_rankings_ccaa.csv",
                                 "Correlación rankings CCAA entre especificaciones"))
    if len(v02) > 0:
        archivos.append(guardar(v02, "V02_sensibilidad_pesos_rankings_sector.csv",
                                 "Correlación rankings sector entre especificaciones"))

    v03 = sensibilidad_umbral(irc, raw)
    if len(v03) > 0:
        archivos.append(guardar(v03, "V03_sensibilidad_umbral_flags.csv",
                                 "Rankings CCAA con p60/p75/p90"))

    v04 = estabilidad_temporal(irc)
    if len(v04) > 0:
        archivos.append(guardar(v04, "V04_estabilidad_temporal_ccaa.csv",
                                 "IRC por subperíodo 2019-21 vs 2022-25"))

    v05 = coherencia_ted(irc)
    if len(v05) > 0:
        archivos.append(guardar(v05, "V05_coherencia_ted_sara.csv",
                                 "Tasa missing TED por grupo flag_B5"))

    # Quality report
    qpath = REPORTS_DIR / "robustez_quality.txt"
    with open(qpath, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("INFORME DE CALIDAD — ROBUSTEZ (Script 08)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 65 + "\n\n")

        f.write(f"Tablas generadas: {len(archivos)}\n")
        for p in archivos:
            f.write(f"  {p.name}\n")

        if len(v01) > 0:
            f.write("\n--- V01: Correlaciones pesos × rankings CCAA ---\n")
            f.write(v01.to_string(index=False))

        if len(v04) > 0:
            corr_col = "corr_spearman_subperiodos"
            if corr_col in v04.columns:
                rho = v04[corr_col].iloc[0]
                f.write(f"\n\n--- V04: Correlación rankings 2019-21 vs 2022-25 ---\n")
                f.write(f"  Spearman rho = {rho:.4f}\n")
                f.write(f"  Interpretación: {'estructural' if abs(rho) >= 0.7 else 'moderada' if abs(rho) >= 0.5 else 'baja'}\n")

        if len(v05) > 0:
            f.write("\n\n--- V05: Coherencia TED ---\n")
            f.write(v05.to_string(index=False))

        f.write("\n\n--- Criterios de robustez ---\n")
        f.write("  PESOS:    robustez si corr(ranking_base, ranking_alt) >= 0.80\n")
        f.write("  UMBRAL:   robustez si corr(ranking_p60, ranking_p90) >= 0.70\n")
        f.write("  TEMPORAL: estructural si corr(2019-21, 2022-25) >= 0.70\n")
        f.write("  TED:      coherente si tasa_missing(flag_B5=alto) > tasa_missing(bajo)\n")

    log.info("Quality report: %s", qpath)
    log.info("=" * 65)
    log.info("Script 08 completado.")
    log.info("Pipeline completo. Todos los outputs listos para el TFG.")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
