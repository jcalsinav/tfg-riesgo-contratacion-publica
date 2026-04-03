"""
script_07_modelos_regresion.py
===============================
Estima los modelos de regresión del apartado 4.d.3 del TFG:
  - MCO con efectos fijos (CCAA, CPV2d, año) y errores Huber-White
  - Logit con variable dependiente irc_alto

Inputs
------
  Nacional/curated/irc_por_nif_ccaa_cpv.parquet           (Script 03)
  Nacional/curated/indicadores_comportamiento_raw.parquet  (Script 02)

Outputs  (Nacional/outputs/modelos/)
-------
  R01_ols_coeficientes.csv
  R02_ols_efectos_fijos_ccaa.csv
  R03_ols_efectos_fijos_cpv2d.csv
  R04_logit_coeficientes.csv
  R05_comparacion_especificaciones.csv   ← robustez: OLS con distintos pesos IRC
  modelos_quality.txt
  logs/script_07.log
"""

from __future__ import annotations
from pathlib import Path
import logging
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

log_path_dir = Path("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos/Nacional/logs")
log_path_dir.mkdir(parents=True, exist_ok=True)

log_path = log_path_dir / "script_07.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

BASE = Path("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos")

IRC_PATH = BASE / "Nacional/curated/irc_por_nif_ccaa_cpv.parquet"
RAW_PATH = BASE / "Nacional/curated/indicadores_comportamiento_raw.parquet"

OUT_DIR     = BASE / "Nacional/outputs/modelos"
REPORTS_DIR = BASE / "Nacional/reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── UTILIDADES ────────────────────────────────────────────────────────────────
def guardar(df: pd.DataFrame, nombre: str, desc: str) -> Path:
    ruta = OUT_DIR / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    log.info("  [OK] %s  (%s filas) — %s", nombre, f"{len(df):,}", desc)
    return ruta


def extraer_coeficientes(resultado, nombre_modelo: str) -> pd.DataFrame:
    """
    Extrae tabla de coeficientes de un resultado statsmodels.
    Incluye: coef, stderr, t/z, p_valor, IC95 inferior y superior.
    """
    try:
        coef   = resultado.params
        stderr = resultado.bse
        tstat  = resultado.tvalues
        pvals  = resultado.pvalues
        ci     = resultado.conf_int()

        df = pd.DataFrame({
            "modelo":    nombre_modelo,
            "variable":  coef.index,
            "coef":      coef.values.round(6),
            "std_err":   stderr.values.round(6),
            "t_z_stat":  tstat.values.round(4),
            "p_valor":   pvals.values.round(6),
            "IC95_inf":  ci.iloc[:, 0].values.round(6),
            "IC95_sup":  ci.iloc[:, 1].values.round(6),
            "sig":       pvals.apply(
                lambda p: "***" if p < 0.001
                else "**" if p < 0.01
                else "*" if p < 0.05
                else "."  if p < 0.1
                else ""
            ).values,
        })
        return df
    except Exception as e:
        log.warning("  Error extrayendo coeficientes: %s", e)
        return pd.DataFrame()


# ── PREPARAR DATOS PARA MODELOS ───────────────────────────────────────────────
def preparar_datos(irc: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la tabla analítica para los modelos:
    irc_base, log_importe, n_contratos, B1_ratio,
    es_pyme (si disponible), cpv2d, ccaa, anio (extraído de fecha_min).
    """
    log.info("Preparando datos para modelos...")

    cols = [
        "adjudicatario_key", "ccaa", "cpv2d",
        "irc_base", "irc_alto", "irc_comp", "irc_iguales",
        "n_contratos", "importe_total",
        "SC_norm", "SB_norm"
    ]
    if "fecha_min" in irc.columns:
        cols.append("fecha_min")
        
    df = irc[cols].copy()

    # Log importe
    df["log_importe"] = np.log(
        pd.to_numeric(df["importe_total"], errors="coerce").clip(lower=1)
    )

    # Año (de fecha_min del grupo si existe)
    if "fecha_min" in df.columns:
        df["fecha_min"] = pd.to_datetime(df["fecha_min"], errors="coerce", utc=True)
        df["anio"] = df["fecha_min"].dt.year.fillna(2022).astype(int)
    else:
        df["anio"] = 2022

    # Unir B1_ratio (proxy num_ofertas) de raw
    if "B1_ratio" in raw.columns:
        df = df.merge(
            raw[["adjudicatario_key","ccaa","cpv2d","B1_ratio"]],
            on=["adjudicatario_key","ccaa","cpv2d"],
            how="left",
        )
    else:
        df["B1_ratio"] = np.nan

    # Filtrar NaN en dependiente
    df = df[df["irc_base"].notna()].copy()

    # Variables categóricas como string para dummies
    df["ccaa"]   = df["ccaa"].astype(str)
    df["cpv2d"]  = df["cpv2d"].astype(str)
    df["anio"]   = df["anio"].astype(str)

    log.info("  Observaciones para modelos: %s", f"{len(df):,}")
    log.info("  Con B1_ratio: %s (%.1f%%)",
             f"{df['B1_ratio'].notna().sum():,}",
             100 * df["B1_ratio"].notna().mean())
    return df


# ── MODELO OLS ────────────────────────────────────────────────────────────────
def estimar_ols(df: pd.DataFrame):
    """
    MCO con efectos fijos de CCAA, CPV2d y año.
    Errores estándar robustos (HC3 = Huber-White).
    Variable dependiente: irc_base.
    """
    log.info("Estimando modelo OLS...")

    # Variables continuas disponibles
    vars_continuas = ["log_importe"]
    if df["B1_ratio"].notna().mean() > 0.1:
        vars_continuas.append("B1_ratio")

    # Fórmula con efectos fijos
    fijos = "C(ccaa) + C(cpv2d) + C(anio)"
    formula = f"irc_base ~ {' + '.join(vars_continuas)} + {fijos}"

    log.info("  Fórmula: %s", formula)

    sub = df[["irc_base","log_importe","B1_ratio","ccaa","cpv2d","anio"]
             ].dropna(subset=["irc_base","log_importe"])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = smf.ols(formula, data=sub).fit(cov_type="HC3")

        log.info("  R²: %.4f | R² adj: %.4f | N: %s",
                 modelo.rsquared, modelo.rsquared_adj, f"{int(modelo.nobs):,}")

        return modelo
    except Exception as e:
        log.error("  Error en OLS: %s", e)
        return None


# ── MODELO LOGIT ──────────────────────────────────────────────────────────────
def estimar_logit(df: pd.DataFrame):
    """
    Logit con irc_alto como variable dependiente binaria.
    Mismas covariables que el OLS.
    """
    log.info("Estimando modelo Logit...")

    sub = df[df["irc_alto"].notna()].copy()
    sub["irc_alto"] = sub["irc_alto"].astype(int)

    vars_continuas = ["log_importe"]
    if sub["B1_ratio"].notna().mean() > 0.1:
        vars_continuas.append("B1_ratio")

    fijos = "C(ccaa) + C(cpv2d) + C(anio)"
    formula = f"irc_alto ~ {' + '.join(vars_continuas)} + {fijos}"

    sub = sub[["irc_alto","log_importe","B1_ratio","ccaa","cpv2d","anio"]
              ].dropna(subset=["irc_alto","log_importe"])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = smf.logit(formula, data=sub).fit(
                method="bfgs", maxiter=200, disp=False
            )

        log.info("  Pseudo-R² (McFadden): %.4f | N: %s",
                 modelo.prsquared, f"{int(modelo.nobs):,}")
        return modelo
    except Exception as e:
        log.error("  Error en Logit: %s", e)
        return None


# ── COMPARACIÓN DE ESPECIFICACIONES ──────────────────────────────────────────
def comparar_especificaciones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estima OLS para las 4 especificaciones del IRC (base, iguales, comp, corp)
    y compara R², coef log_importe y coef B1_ratio.
    Objetivo: validación de robustez (apartado 4.e).
    """
    log.info("Comparando especificaciones del IRC...")

    specs = {
        "irc_base":    "wC=0.4, wB=0.6 (base)",
        "irc_iguales": "wC=0.5, wB=0.5",
        "irc_comp":    "wC=0.0, wB=1.0 (solo comportamiento)",
        "irc_corp":    "wC=1.0, wB=0.0 (solo corporativo)",
    }

    rows = []
    for col, desc in specs.items():
        if col not in df.columns:
            continue
        sub = df[df[col].notna() & df["log_importe"].notna()].copy()
        if len(sub) < 50:
            continue

        formula = f"{col} ~ log_importe + C(ccaa) + C(cpv2d) + C(anio)"
        if sub["B1_ratio"].notna().mean() > 0.1:
            formula = f"{col} ~ log_importe + B1_ratio + C(ccaa) + C(cpv2d) + C(anio)"

        sub = sub[["log_importe","B1_ratio","ccaa","cpv2d","anio",col]
                  ].dropna(subset=["log_importe",col])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.ols(formula, data=sub).fit(cov_type="HC3")
            row = {
                "especificacion":   col,
                "descripcion":      desc,
                "n":                int(m.nobs),
                "R2":               round(m.rsquared, 4),
                "R2_adj":           round(m.rsquared_adj, 4),
                "coef_log_importe": round(m.params.get("log_importe", np.nan), 6),
                "p_log_importe":    round(m.pvalues.get("log_importe", np.nan), 6),
                "coef_B1_ratio":    round(m.params.get("B1_ratio", np.nan), 6),
                "p_B1_ratio":       round(m.pvalues.get("B1_ratio", np.nan), 6),
            }
            rows.append(row)
            log.info("  %s: R²=%.4f", col, m.rsquared)
        except Exception as e:
            log.warning("  %s: error — %s", col, e)

    return pd.DataFrame(rows)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("SCRIPT 07 — Modelos de regresión")
    log.info("Inicio: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    if not HAS_STATSMODELS:
        log.error("statsmodels no instalado. Ejecuta: pip install statsmodels")
        raise ImportError("statsmodels requerido para script_07")

    if not IRC_PATH.exists():
        raise FileNotFoundError(f"IRC no encontrado: {IRC_PATH}")

    irc = pd.read_parquet(IRC_PATH)
    raw = pd.read_parquet(RAW_PATH) if RAW_PATH.exists() else pd.DataFrame()

    df = preparar_datos(irc, raw)

    archivos = []

    # OLS
    ols = estimar_ols(df)
    if ols is not None:
        coef_ols = extraer_coeficientes(ols, "OLS_base")
        # Separar efectos fijos de coeficientes de interés
        vars_interes = coef_ols[~coef_ols["variable"].str.startswith("C(")]
        ef_ccaa      = coef_ols[coef_ols["variable"].str.contains("C\\(ccaa\\)")]
        ef_cpv2d     = coef_ols[coef_ols["variable"].str.contains("C\\(cpv2d\\)")]

        archivos.append(guardar(vars_interes, "R01_ols_coeficientes.csv",
                                 "OLS coeficientes variables de interés"))
        archivos.append(guardar(ef_ccaa, "R02_ols_efectos_fijos_ccaa.csv",
                                 "OLS efectos fijos CCAA"))
        archivos.append(guardar(ef_cpv2d, "R03_ols_efectos_fijos_cpv2d.csv",
                                 "OLS efectos fijos CPV2d"))

    # Logit
    logit = estimar_logit(df)
    if logit is not None:
        coef_logit = extraer_coeficientes(logit, "Logit_irc_alto")
        vars_logit = coef_logit[~coef_logit["variable"].str.startswith("C(")]
        archivos.append(guardar(vars_logit, "R04_logit_coeficientes.csv",
                                 "Logit coeficientes variables de interés"))

    # Robustez: comparar especificaciones
    comp = comparar_especificaciones(df)
    if len(comp) > 0:
        archivos.append(guardar(comp, "R05_comparacion_especificaciones.csv",
                                 "Comparación R² entre especificaciones IRC"))

    # Quality report
    qpath = REPORTS_DIR / "modelos_quality.txt"
    with open(qpath, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("INFORME DE CALIDAD — MODELOS REGRESIÓN (Script 07)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 65 + "\n\n")
        if ols is not None:
            f.write(f"OLS — R²: {ols.rsquared:.4f} | R² adj: {ols.rsquared_adj:.4f}"
                    f" | N: {int(ols.nobs):,}\n")
            f.write(f"OLS — AIC: {ols.aic:.2f} | BIC: {ols.bic:.2f}\n\n")
        if logit is not None:
            f.write(f"Logit — Pseudo-R² (McFadden): {logit.prsquared:.4f}"
                    f" | N: {int(logit.nobs):,}\n\n")
        f.write(f"Tablas generadas: {len(archivos)}\n")
        for p in archivos:
            f.write(f"  {p.name}\n")

    log.info("Quality report: %s", qpath)
    log.info("=" * 65)
    log.info("Script 07 completado. SIGUIENTE: script_08_validaciones_robustez.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
