"""
script_04_tablas_descriptivas.py
=================================
Genera todas las tablas del apartado 5.a (Panorama general) del TFG.

Inputs
------
  Nacional/curated/irc_por_nif_ccaa_cpv.parquet          (Script 03)
  Nacional/curated/indicadores_comportamiento_raw.parquet (Script 02)
  Nacional/BORME/curated/flags_corporativos.parquet       (Script 01)
  + Curados por fuente (para estadísticos de volumen)

Outputs  (todos en Nacional/outputs/descriptivos/)
-------
  T01_volumen_por_anio_procedimiento.csv
  T02_volumen_por_ccaa_fuente.csv
  T03_distribucion_importes_percentiles.csv
  T04_cobertura_campos_clave.csv
  T05_top10_adjudicatarios_importe.csv     ← sin nombre, solo rank + valor
  T06_top10_adjudicatarios_contratos.csv   ← ídem
  T07_distribucion_irc_cuartil_ccaa.csv
  T08_distribucion_irc_cuartil_sector.csv
  T09_flags_corporativos_resumen.csv
  T10_flags_comportamiento_resumen.csv
  descriptivos_quality.txt
  logs/script_04.log
"""

from __future__ import annotations
from pathlib import Path
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# ── RUTAS ─────────────────────────────────────────────────────────────────────
BASE = Path("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos")

IRC_PATH   = BASE / "Nacional/curated/irc_por_nif_ccaa_cpv.parquet"
RAW_PATH   = BASE / "Nacional/curated/indicadores_comportamiento_raw.parquet"
FLAGS_CORP = BASE / "Nacional/BORME/curated/flags_corporativos.parquet"

CURADOS = {
    "placsp":    BASE / "Nacional/PLACSP/curated/placsp_2019_2025_curated.parquet",
    "andalucia": BASE / "CCAA/Andalucia/curated/andalucia_2019_2025_curated.parquet",
    "cataluna":  BASE / "CCAA/Cataluna/curated/cataluna_2019_2025_curated_pipeline.parquet",
    "madrid":    BASE / "CCAA/Comunidad_Madrid/curated/madrid_cam_2019_2025_curated.parquet",
    "euskadi":   BASE / "CCAA/Euskadi/curated/euskadi_2019_2025_curated.parquet",
    "galicia":   BASE / "CCAA/Galicia/curated/galicia_2019_2025_curated.parquet",
    "valencia":  BASE / "CCAA/Valencia/curated/valencia_2019_2025_regcon_consolidado.parquet",
}

FUENTE_A_CCAA = {
    "placsp": "NAC", "andalucia": "AN", "cataluna": "CT",
    "madrid": "MD", "euskadi": "PV",
    "galicia": "GA", "valencia": "VC",
}

OUT_DIR     = BASE / "Nacional/outputs/descriptivos"
REPORTS_DIR = BASE / "Nacional/reports"
LOGS_DIR    = BASE / "Nacional/logs"
for d in [OUT_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_path = LOGS_DIR / "script_04.log"
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
def guardar(df: pd.DataFrame, nombre: str, descripcion: str) -> Path:
    ruta = OUT_DIR / nombre
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    log.info("  [OK] %s  (%s filas) — %s", nombre, f"{len(df):,}", descripcion)
    return ruta


def _find_col(df, candidates):
    """Detección case-insensitive y sin acentos de columnas."""
    import unicodedata
    def clean(s):
        s = unicodedata.normalize('NFKD', str(s)).encode('ASCII', 'ignore').decode('utf-8')
        return s.lower().replace(" ", "_").replace(".", "_")
        
    col_map = {clean(c): c for c in df.columns}
    for cand in candidates:
        key = clean(cand)
        if key in col_map:
            return col_map[key]
    return None


def _parse_importe(serie: pd.Series) -> pd.Series:
    """
    Parsea importes numéricos o en formato europeo de texto ('18.148,79').
    Necesario para Madrid CAM, cuyo curado almacena importes como strings europeos.
    """
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)
    s = serie.astype(str).str.strip()
    es_europeo = (
        s.str.contains(r"\d\.\d{3}", na=False) |
        s.str.contains(r"\d,\d{1,2}$", na=False)
    )
    resultado = pd.Series(np.nan, index=serie.index, dtype=float)
    if es_europeo.any():
        convertido = (s[es_europeo]
                      .str.replace(".", "", regex=False)
                      .str.replace(",", ".", regex=False))
        resultado[es_europeo] = pd.to_numeric(convertido, errors="coerce")
    if (~es_europeo).any():
        resultado[~es_europeo] = pd.to_numeric(s[~es_europeo], errors="coerce")
    return resultado


def percentiles_importe(serie: pd.Series) -> dict:
    s = pd.to_numeric(serie, errors="coerce").dropna()
    s = s[s > 0]
    if len(s) == 0:
        return {}
    return {
        "n":    len(s),
        "p05":  s.quantile(0.05),
        "p25":  s.quantile(0.25),
        "p50":  s.quantile(0.50),
        "p75":  s.quantile(0.75),
        "p95":  s.quantile(0.95),
        "max":  s.max(),
        "media": s.mean(),
        "suma": s.sum(),
    }


# ── PASO 1: Cargar corpus liviano (solo columnas necesarias) ──────────────────
def cargar_corpus_ligero() -> pd.DataFrame:
    """
    Carga solo las columnas necesarias para las tablas descriptivas.
    Mucho más rápido que cargar todo el corpus del Script 02.
    """
    log.info("Cargando corpus ligero...")
    partes = []

    for fuente, ruta in CURADOS.items():
        if not ruta.exists():
            log.warning("  Omitido: %s", ruta)
            continue

        df = pd.read_parquet(ruta)

        tmp = pd.DataFrame()

        # Clave adjudicatario (para conteos únicos)
        col_nif = _find_col(df, [
            "adjudicatario_nif", "nif_adjudicatario", "nif",
            "nif_adj", "adjudicatario nif", "nif del adjudicatario",
            "nif_empresa", "empresa_nif", "cif",
            "cif_nif_enmascarado",                          # Valencia REGCON
        ])
        tmp["adj_key"] = df[col_nif].astype(str) if col_nif else "desconocido"

        # Importe
        col_imp = _find_col(df, [
            "importe_adjudicacion", "importe_licitacion", "importe",
            "importe de adjudicacion", "importe_adj", "precio_adjudicacion",
            "importe adjudicacion", "valor_adjudicacion",
            "imp_adjud_sin_iva",                            # Valencia REGCON (preferido)
            "imp_total_adjud",                             # Valencia REGCON (alternativo)
            "importe_licit_sin_iva",                       # Valencia REGCON (licitación)
        ])
        tmp["importe"] = _parse_importe(df[col_imp]) if col_imp else np.nan

        # Fecha / año
        col_fecha = _find_col(df, [
            "fecha_referencia", "fecha_adjudicacion", "fecha_publicacion",
            "fecha_formalizacion",                          # Valencia REGCON
            "fecha_publ_adj_form_docv",                    # Valencia REGCON (alternativo)
        ])
        if col_fecha:
            fecha = pd.to_datetime(df[col_fecha], errors="coerce", utc=True)
            tmp["anio"] = fecha.dt.year
        else:
            tmp["anio"] = np.nan

        # Procedimiento
        col_proc = _find_col(df, [
            "procedimiento_code", "procedimiento",         # Valencia REGCON tiene "PROCEDIMIENTO"
        ])
        tmp["procedimiento_code"] = df[col_proc].astype(str) if col_proc else "NA"

        # Tipo contrato
        col_tipo = _find_col(df, [
            "tipo_contrato", "tipo_de_contrato",
            "clase_de_contrato",                            # Valencia REGCON
            "tipo",                                         # Valencia REGCON (campo TIPO)
        ])
        tmp["tipo_contrato"] = df[col_tipo].astype(str) if col_tipo else "NA"

        # Contrato menor
        if "es_menor" in df.columns:
            tmp["es_menor"] = df["es_menor"].astype(bool)
        elif "es_menor_explicito" in df.columns:
            tmp["es_menor"] = df["es_menor_explicito"].astype(bool)
        else:
            tmp["es_menor"] = False

        # num_ofertas
        col_of = next(
            (c for c in ["num_ofertas","numero_ofertas","Nº de ofertas"]
             if c in df.columns), None
        )
        tmp["num_ofertas"] = pd.to_numeric(df[col_of], errors="coerce") if col_of else np.nan

        # NIF informado
        tmp["nif_informado"] = tmp["adj_key"].notna() & ~tmp["adj_key"].isin(
            ["nan","NAN","None","","desconocido"]
        )

        tmp["fuente"] = fuente
        tmp["ccaa"]   = FUENTE_A_CCAA.get(fuente, "NAC")
        partes.append(tmp)

    corpus = pd.concat(partes, ignore_index=True)
    log.info("  Corpus ligero: %s filas", f"{len(corpus):,}")
    return corpus


# ── T01: Volumen por año y tipo de procedimiento ──────────────────────────────
def tabla_T01(corpus: pd.DataFrame) -> pd.DataFrame:
    log.info("T01: volumen por año y procedimiento...")

    MAPA_PROC = {
        "1": "Abierto", "2": "Restringido", "3": "Neg. c/publicidad",
        "4": "Neg. s/publicidad", "6": "Sin declarar (cód.6)",
        "7": "Contrato menor", "9": "Sistema dinámico",
        "100": "Acuerdo marco", "999": "Otros", "NA": "No disponible",
    }

    sub = corpus[corpus["anio"].between(2019, 2025)].copy()
    sub["proc_label"] = sub["procedimiento_code"].map(MAPA_PROC).fillna("Otros")

    t01 = (
        sub.groupby(["anio","proc_label"])
        .agg(
            n_contratos=("adj_key","count"),
            importe_total=("importe","sum"),
            importe_mediana=("importe","median"),
        )
        .reset_index()
    )
    t01["anio"] = t01["anio"].astype(int)
    t01 = t01.sort_values(["anio","n_contratos"], ascending=[True, False])
    return t01


# ── T02: Volumen por CCAA y fuente ────────────────────────────────────────────
def tabla_T02(corpus: pd.DataFrame) -> pd.DataFrame:
    log.info("T02: volumen por CCAA y fuente...")

    # No se filtra por año: corpus ya viene filtrado 2019-2025 desde script_02.
    # El filtro between() descartaba filas con anio=NaN (Madrid CAM, Catalunya).
    # dropna=False para no descartar ccaa/fuente que puedan ser NaN.
    log.info("  Distribucion ccaa en corpus: %s",
             corpus["ccaa"].value_counts(dropna=False).to_dict())
    t02 = (
        corpus.groupby(["ccaa","fuente"], dropna=False)
        .agg(
            n_contratos   =("adj_key","count"),
            n_menores     =("es_menor","sum"),
            importe_total =("importe","sum"),
            n_adj_unicos  =("adj_key","nunique"),
        )
        .reset_index()
    )
    t02["pct_menores"] = (t02["n_menores"] / t02["n_contratos"] * 100).round(1)
    t02 = t02.sort_values("n_contratos", ascending=False)
    return t02


# ── T03: Distribución de importes por percentiles ─────────────────────────────
def tabla_T03(corpus: pd.DataFrame) -> pd.DataFrame:
    log.info("T03: distribución importes por percentiles...")

    grupos = {
        "Global": corpus,
        "Menores":    corpus[corpus["es_menor"]],
        "No menores": corpus[~corpus["es_menor"]],
    }
    # Por CCAA
    for ccaa in sorted(corpus["ccaa"].dropna().unique()):
        grupos[f"CCAA {ccaa}"] = corpus[corpus["ccaa"] == ccaa]

    filas = []
    for nombre, sub in grupos.items():
        p = percentiles_importe(sub["importe"])
        if p:
            filas.append({"grupo": nombre, **p})

    return pd.DataFrame(filas)


# ── T04: Cobertura de campos clave ────────────────────────────────────────────
def tabla_T04(corpus: pd.DataFrame) -> pd.DataFrame:
    log.info("T04: cobertura campos clave...")

    n = len(corpus)
    filas = []
    for fuente in sorted(corpus["fuente"].unique()):
        sub = corpus[corpus["fuente"] == fuente]
        ns  = len(sub)
        filas.append({
            "fuente":          fuente,
            "n_registros":     ns,
            "pct_nif_informado":   round(100 * sub["nif_informado"].mean(), 1),
            "pct_importe_no_nulo": round(100 * sub["importe"].notna().mean(), 1),
            "pct_num_ofertas":     round(100 * sub["num_ofertas"].notna().mean(), 1),
            "pct_es_menor_true":   round(100 * sub["es_menor"].mean(), 1),
        })
    # Total
    filas.append({
        "fuente": "TOTAL",
        "n_registros":     n,
        "pct_nif_informado":   round(100 * corpus["nif_informado"].mean(), 1),
        "pct_importe_no_nulo": round(100 * corpus["importe"].notna().mean(), 1),
        "pct_num_ofertas":     round(100 * corpus["num_ofertas"].notna().mean(), 1),
        "pct_es_menor_true":   round(100 * corpus["es_menor"].mean(), 1),
    })
    return pd.DataFrame(filas)


# ── T05 / T06: Top 10 adjudicatarios (sin nombre) ────────────────────────────
def tablas_T05_T06(corpus: pd.DataFrame):
    log.info("T05/T06: top 10 adjudicatarios...")

    sub = corpus[corpus["nif_informado"] & corpus["importe"].notna()].copy()

    # T05 — por importe acumulado
    t05 = (
        sub.groupby("adj_key")
        .agg(importe_acumulado=("importe","sum"), n_contratos=("adj_key","count"))
        .nlargest(10, "importe_acumulado")
        .reset_index(drop=True)
    )
    t05.index = t05.index + 1
    t05.index.name = "rank"
    t05 = t05.reset_index()[["rank","importe_acumulado","n_contratos"]]

    # T06 — por recuento de contratos
    t06 = (
        sub.groupby("adj_key")
        .agg(n_contratos=("adj_key","count"), importe_acumulado=("importe","sum"))
        .nlargest(10, "n_contratos")
        .reset_index(drop=True)
    )
    t06.index = t06.index + 1
    t06.index.name = "rank"
    t06 = t06.reset_index()[["rank","n_contratos","importe_acumulado"]]

    return t05, t06


# ── T07 / T08: Distribución IRC por cuartil ──────────────────────────────────
def tablas_T07_T08(irc: pd.DataFrame):
    log.info("T07/T08: distribución IRC por cuartil...")

    sub = irc[irc["irc_cuartil"].notna()].copy()

    # T07 — por CCAA
    t07 = (
        sub.groupby(["ccaa","irc_cuartil"])
        .agg(
            n_grupos     =("adjudicatario_key","count"),
            irc_mediana  =("irc_base","median"),
            importe_total=("importe_total","sum"),
        )
        .reset_index()
    )
    t07["irc_cuartil"] = t07["irc_cuartil"].astype(int)
    t07 = t07.sort_values(["ccaa","irc_cuartil"])

    # T08 — por sector CPV2d
    t08 = (
        sub.groupby(["cpv2d","irc_cuartil"])
        .agg(
            n_grupos     =("adjudicatario_key","count"),
            irc_mediana  =("irc_base","median"),
            importe_total=("importe_total","sum"),
        )
        .reset_index()
    )
    t08["irc_cuartil"] = t08["irc_cuartil"].astype(int)
    # Solo top 20 sectores por importe total
    top_cpv = (
        sub.groupby("cpv2d")["importe_total"].sum()
        .nlargest(20).index.tolist()
    )
    t08 = t08[t08["cpv2d"].isin(top_cpv)]
    t08 = t08.sort_values(["cpv2d","irc_cuartil"])

    return t07, t08


# ── T09: Resumen flags corporativos ──────────────────────────────────────────
def tabla_T09(flags_corp: pd.DataFrame) -> pd.DataFrame:
    log.info("T09: resumen flags corporativos...")

    n = len(flags_corp)
    filas = []
    for flag, desc in [
        ("F1", "Empresa recién constituida (<6 meses)"),
        ("F2", "Capital social bajo (<10K€) + importe alto (>100K€)"),
        ("F3", "Administradores compartidos (mismo órgano)"),
        ("F4", "Disolución <12 meses post-adjudicación"),
        ("F5", "Empresa en situación concursal"),
    ]:
        if flag not in flags_corp.columns:
            continue
        n_flag = int(flags_corp[flag].sum())
        filas.append({
            "flag":        flag,
            "descripcion": desc,
            "n_activos":   n_flag,
            "pct_total":   round(100 * n_flag / max(n, 1), 2),
            "pct_con_borme": round(
                100 * n_flag / max(int(flags_corp.get("borme_cobertura", pd.Series()).sum()), 1),
                2
            ),
        })

    # Score corporativo
    if "score_corporativo" in flags_corp.columns:
        dist = flags_corp["score_corporativo"].value_counts().sort_index()
        for sc, cnt in dist.items():
            filas.append({
                "flag":        f"Score={int(sc)}",
                "descripcion": f"Adjudicatarios con {int(sc)} flags activos",
                "n_activos":   int(cnt),
                "pct_total":   round(100 * cnt / max(n, 1), 2),
                "pct_con_borme": "",
            })

    return pd.DataFrame(filas)


# ── T10: Resumen flags de comportamiento ─────────────────────────────────────
def tabla_T10(raw: pd.DataFrame) -> pd.DataFrame:
    log.info("T10: resumen flags de comportamiento...")

    n = len(raw)
    INDICADORES = {
        "B1_ratio":     ("B1", "Proporción oferta única"),
        "B2_hhi_medio": ("B2", "HHI por órgano (concentración)"),
        "B3_racha_max": ("B3", "Racha máxima consecutiva"),
        "B4_ratio":     ("B4", "Clustering umbral contrato menor"),
        "B5_ratio":     ("B5", "Clustering umbral SARA"),
        "B6_ratio":     ("B6", "Procedimientos excepcionales"),
    }

    filas = []
    for col, (bx, desc) in INDICADORES.items():
        if col not in raw.columns:
            continue
        s = raw[col].dropna()
        filas.append({
            "indicador":   bx,
            "descripcion": desc,
            "cobertura_n": len(s),
            "cobertura_pct": round(100 * len(s) / max(n, 1), 1),
            "media":       round(s.mean(), 4),
            "p50":         round(s.quantile(0.50), 4),
            "p75":         round(s.quantile(0.75), 4),
            "max":         round(s.max(), 4),
        })

    return pd.DataFrame(filas)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("SCRIPT 04 — Tablas descriptivas (apartado 5.a)")
    log.info("Inicio: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    # Cargar inputs
    corpus = cargar_corpus_ligero()

    irc = pd.read_parquet(IRC_PATH) if IRC_PATH.exists() else None
    if irc is None:
        log.warning("IRC no disponible. T07/T08 no se generarán.")

    raw_comp = pd.read_parquet(RAW_PATH) if RAW_PATH.exists() else None
    if raw_comp is None:
        log.warning("Raw comportamiento no disponible. T10 no se generará.")

    flags_corp = pd.read_parquet(FLAGS_CORP) if FLAGS_CORP.exists() else None
    if flags_corp is None:
        log.warning("Flags corporativos no disponibles. T09 no se generará.")

    # Generar tablas
    archivos = []

    t01 = tabla_T01(corpus)
    archivos.append(guardar(t01, "T01_volumen_por_anio_procedimiento.csv",
                             "Volumen contratos por año y tipo procedimiento"))

    t02 = tabla_T02(corpus)
    archivos.append(guardar(t02, "T02_volumen_por_ccaa_fuente.csv",
                             "Volumen contratos por CCAA y fuente"))

    t03 = tabla_T03(corpus)
    archivos.append(guardar(t03, "T03_distribucion_importes_percentiles.csv",
                             "Distribución importes por grupo"))

    t04 = tabla_T04(corpus)
    archivos.append(guardar(t04, "T04_cobertura_campos_clave.csv",
                             "Cobertura campos clave por fuente"))

    t05, t06 = tablas_T05_T06(corpus)
    archivos.append(guardar(t05, "T05_top10_adjudicatarios_importe.csv",
                             "Top 10 por importe acumulado (sin nombre)"))
    archivos.append(guardar(t06, "T06_top10_adjudicatarios_contratos.csv",
                             "Top 10 por nº contratos (sin nombre)"))

    if irc is not None:
        t07, t08 = tablas_T07_T08(irc)
        archivos.append(guardar(t07, "T07_distribucion_irc_cuartil_ccaa.csv",
                                 "IRC por cuartil × CCAA"))
        archivos.append(guardar(t08, "T08_distribucion_irc_cuartil_sector.csv",
                                 "IRC por cuartil × sector CPV2d"))

    if flags_corp is not None:
        t09 = tabla_T09(flags_corp)
        archivos.append(guardar(t09, "T09_flags_corporativos_resumen.csv",
                                 "Resumen activación flags corporativos"))

    if raw_comp is not None:
        t10 = tabla_T10(raw_comp)
        archivos.append(guardar(t10, "T10_flags_comportamiento_resumen.csv",
                                 "Estadísticos indicadores de comportamiento"))

    # Quality report
    qpath = REPORTS_DIR / "descriptivos_quality.txt"
    with open(qpath, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("INFORME DE CALIDAD — TABLAS DESCRIPTIVAS (Script 04)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Corpus total: {len(corpus):,} filas\n")
        f.write(f"Tablas generadas: {len(archivos)}\n\n")
        for p in archivos:
            f.write(f"  {p.name}\n")

    log.info("Quality report: %s", qpath)
    log.info("=" * 65)
    log.info("Script 04 completado. SIGUIENTE: script_05_mapa_territorio_sector.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
