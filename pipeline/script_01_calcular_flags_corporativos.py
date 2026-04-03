"""
script_01_calcular_flags_corporativos.py  (v2 — post-diagnóstico)
==================================================================
Cruza el dataset curado de contratación 2019-2025 con el BORME
para calcular los cinco indicadores corporativos (F1-F5) por
adjudicatario.

Hallazgos del diagnóstico que condicionan el diseño:
  - BORME no tiene columna NIF/CIF → join por empresa_norm
  - capital_euros:      columna directa, 76.8% nulos (solo en actos de capital)
  - fecha_constitucion: dtype object, formatos irregulares (dd.mm.yy),
                        necesita parseo con errors='coerce'
  - actos:              pipe-separated ("Constitución|Nombramientos|...")
  - borme_cargos:       tampoco tiene NIF → join por empresa_norm

Riesgo principal: colisiones de nombre en el join.
  → Se detectan y marcan en columna 'join_colision_nombre'
  → Los flags de NIFs con colisión se marcan como 'indeterminado'
    en 'flags_fiabilidad' para documentarlos en el TFG.

Outputs
-------
  curated/flags_corporativos.parquet
  reports/flags_corporativos_quality.txt
  logs/script_01.log
"""

from __future__ import annotations
from pathlib import Path
import logging
import re
import unicodedata
from datetime import datetime

import pandas as pd
import numpy as np

# ── RUTAS ─────────────────────────────────────────────────────────────────────
BASE = Path("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos")

BORME_EMPRESAS = BASE / "Nacional/BORME/raw/borme_empresas_pub.parquet"
BORME_CARGOS   = BASE / "Nacional/BORME/raw/borme_cargos_pub.parquet"

CURADOS = {
    "placsp":    BASE / "Nacional/PLACSP/curated/placsp_2019_2025_curated.parquet",
    "andalucia": BASE / "CCAA/Andalucia/curated/andalucia_2019_2025_curated.parquet",
    "cataluna":  BASE / "CCAA/Cataluna/curated/cataluna_2019_2025_curated_pipeline.parquet",
    "madrid":    BASE / "CCAA/Comunidad_Madrid/curated/madrid_cam_2019_2025_curated.parquet",
    "euskadi":   BASE / "CCAA/Euskadi/curated/euskadi_2019_2025_curated.parquet",
    "galicia":   BASE / "CCAA/Galicia/curated/galicia_2019_2025_curated.parquet",
    "valencia":  BASE / "CCAA/Valencia/curated/valencia_2019_2025_regcon_consolidado.parquet",
}

OUT_DIR     = BASE / "Nacional/BORME/curated"
REPORTS_DIR = BASE / "Nacional/BORME/reports"
LOGS_DIR    = BASE / "Nacional/BORME/logs"

for d in [OUT_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_path = LOGS_DIR / "script_01.log"
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
FORMAS_JURIDICAS = re.compile(
    r"\b(s\.?l\.?u?\.?|s\.?a\.?u?\.?|s\.?l\.?l\.?|s\.?l\.?p\.?|s\.?c\.?|"
    r"s\.?a\.?t\.?|s\.?c\.?o\.?o\.?p\.?|s\.?l\.?n\.?e\.?|coop\.?|"
    r"sociedad limitada|sociedad anonima|sl|sa|sau|slu)\b",
    re.IGNORECASE,
)

UMBRAL_F1_DIAS    = 180      # < 6 meses entre constitución y 1ª adjudicación
UMBRAL_F2_CAPITAL = 10_000   # capital < 10K€
UMBRAL_F2_IMPORTE = 100_000  # importe acumulado > 100K€
UMBRAL_F4_DIAS    = 365      # disolución < 12 meses tras última adjudicación


# ── UTILIDADES ────────────────────────────────────────────────────────────────
def normalizar_nif(serie: pd.Series) -> pd.Series:
    return (
        serie.astype(str)
        .str.upper()
        .str.replace(r"[\s\-\.]", "", regex=True)
        .str.strip()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan, "NULL": np.nan})
    )


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Busca la primera columna que coincida con algún candidato,
    de forma case-insensitive y tolerante a espacios/guiones bajos y acentos.
    Registra las columnas disponibles si no hay match, para facilitar debug.
    """
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
    Parsea importes que pueden venir como:
    - Numérico directo (la mayoría de fuentes)
    - Texto formato europeo: '18.148,79' o '1.234.567,89'  ← Madrid CAM
    - Texto formato anglosajón: '18148.79'
    Devuelve siempre float64.
    """
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)

    s = serie.astype(str).str.strip()

    # Detectar formato europeo: tiene punto como sep. miles Y coma como decimal
    # Patrón: dígito + punto + exactamente 3 dígitos  →  separador de miles
    es_europeo = (
        s.str.contains(r"\d\.\d{3}", na=False) |
        s.str.contains(r"\d,\d{1,2}$", na=False)
    )

    resultado = pd.Series(np.nan, index=serie.index, dtype=float)

    if es_europeo.any():
        convertido = (
            s[es_europeo]
            .str.replace(".", "", regex=False)   # quitar sep. miles
            .str.replace(",", ".", regex=False)  # coma decimal → punto
        )
        resultado[es_europeo] = pd.to_numeric(convertido, errors="coerce")

    if (~es_europeo).any():
        resultado[~es_europeo] = pd.to_numeric(s[~es_europeo], errors="coerce")

    return resultado


def normalizar_nombre_empresa(serie: pd.Series) -> pd.Series:
    """
    Normalización para join por nombre:
    1. Minúsculas + quitar acentos
    2. Eliminar formas jurídicas (SL, SA, SAU…)
    3. Colapsar espacios y strip
    """
    s = serie.astype(str).str.lower().str.strip()
    s = s.apply(
        lambda x: unicodedata.normalize("NFKD", x)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    s = s.apply(lambda x: FORMAS_JURIDICAS.sub("", x))
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.replace({"nan": np.nan, "": np.nan})
    return s


def parsear_fecha_constitucion(serie: pd.Series) -> pd.Series:
    """
    Parsea fecha_constitucion del BORME (dtype object, formatos dd.mm.yy irregulares).
    """
    s = serie.astype(str).str.strip()
    s = s.where(s.str.len() >= 6, other=np.nan)
    s = s.where(~s.str.fullmatch(r"[.\s]+").fillna(False), other=np.nan)

    parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")

    # Corregir años > actual+1 (formato yy interpretado como 20xx)
    anio_actual = datetime.now().year
    mask_futuro = parsed.dt.year > (anio_actual + 1)
    # En lugar de restar 100 años (overflow), marcar como NaT las fechas futuras
    parsed = parsed.where(~mask_futuro, other=pd.NaT)
    return parsed


# ── PASO 1: Cargar corpus curado ──────────────────────────────────────────────
def cargar_corpus() -> pd.DataFrame:
    partes = []

    for nombre, ruta in CURADOS.items():
        if not ruta.exists():
            log.warning("Curado no encontrado, se omite: %s", ruta)
            continue

        log.info("Cargando: %s", nombre)
        df = pd.read_parquet(ruta)

        col_nif = _find_col(df, [
            "adjudicatario_nif", "nif_adjudicatario", "nif",
            "nif_adj", "adjudicatario nif", "nif del adjudicatario",
            "nif_empresa", "empresa_nif", "cif", "adjudicatario_cif",
            "cif_nif_enmascarado",                          # Valencia REGCON
        ])
        col_nombre = _find_col(df, [
            "adjudicatario_nombre", "adjudicatario", "razon_social_adjudicatario",
            "nombre_adjudicatario", "empresa", "razon_social", "nombre_empresa",
            "adjudicatario nombre", "nombre del adjudicatario",
            "nombre_o_razon_social",                        # Valencia REGCON
        ])
        col_organo_nif = _find_col(df, [
            "organo_nif", "nif_organo", "nif_organo_contratante",
            "organo_contratante_nif", "nif del organo",
        ])
        col_importe = _find_col(df, [
            "importe_adjudicacion", "importe_licitacion", "importe",
            "importe de adjudicacion", "importe_adj", "precio_adjudicacion",
            "importe adjudicacion", "valor_adjudicacion",
            "imp_adjud_sin_iva",                            # Valencia REGCON (preferido)
            "imp_total_adjud",                             # Valencia REGCON (alternativo)
            "importe_licit_sin_iva",                       # Valencia REGCON (licitación)
        ])
        col_fecha = _find_col(df, [
            "fecha_referencia", "fecha_adjudicacion", "fecha_publicacion",
            "fecha_formalizacion", "fecha",
            "fecha_formalizacion",                         # Valencia REGCON
            "fecha_publ_adj_form_docv",                    # Valencia REGCON (alternativo)
        ])

        if col_nif is None and col_nombre is None:
            log.warning("  %s: sin NIF ni nombre adjudicatario, se omite.", nombre)
            log.warning("    Columnas disponibles: %s", list(df.columns))
            continue

        tmp = pd.DataFrame()
        tmp["adjudicatario_nif"]    = normalizar_nif(df[col_nif]) if col_nif else np.nan
        tmp["adjudicatario_nombre"] = df[col_nombre].astype(str) if col_nombre else np.nan
        tmp["organo_nif"]           = normalizar_nif(df[col_organo_nif]) if col_organo_nif else np.nan
        tmp["importe_adjudicacion"] = _parse_importe(df[col_importe]) if col_importe else np.nan
        tmp["_fecha"]               = pd.to_datetime(df[col_fecha], errors="coerce", utc=True) if col_fecha else pd.NaT
        tmp["_fuente"]              = nombre

        partes.append(tmp)

    corpus = pd.concat(partes, ignore_index=True)
    corpus["adjudicatario_nombre_norm"] = normalizar_nombre_empresa(
        corpus["adjudicatario_nombre"]
    )

    sin_id = corpus["adjudicatario_nif"].isna() & corpus["adjudicatario_nombre_norm"].isna()
    n_desc = int(sin_id.sum())
    corpus = corpus[~sin_id].copy()

    log.info("Corpus total: %s filas | Descartadas (sin ID): %s",
             f"{len(corpus):,}", f"{n_desc:,}")
    log.info("NIFs únicos: %s | Nombres norm únicos: %s",
             f"{corpus['adjudicatario_nif'].nunique():,}",
             f"{corpus['adjudicatario_nombre_norm'].nunique():,}")
    return corpus


# ── PASO 2: Tabla base por adjudicatario ─────────────────────────────────────
def construir_tabla_nif(corpus: pd.DataFrame) -> pd.DataFrame:
    corpus["_clave"] = corpus["adjudicatario_nif"].fillna(
        corpus["adjudicatario_nombre_norm"]
    )
    corpus = corpus[corpus["_clave"].notna()].copy()

    agg = corpus.groupby("_clave").agg(
        adjudicatario_nif         =("adjudicatario_nif",          "first"),
        adjudicatario_nombre      =("adjudicatario_nombre",        "first"),
        adjudicatario_nombre_norm =("adjudicatario_nombre_norm",   "first"),
        fecha_primera_adj         =("_fecha",                      "min"),
        fecha_ultima_adj          =("_fecha",                      "max"),
        importe_acumulado         =("importe_adjudicacion",        "sum"),
        n_contratos               =("_clave",                      "count"),
    ).reset_index(drop=True)

    agg["_solo_nombre"] = agg["adjudicatario_nif"].isna()

    log.info("Tabla base: %s adjudicatarios (sin NIF: %s)",
             f"{len(agg):,}", f"{int(agg['_solo_nombre'].sum()):,}")
    return agg


# ── PASO 3: Preparar BORME empresas ──────────────────────────────────────────
def preparar_borme_empresas() -> pd.DataFrame:
    log.info("Cargando BORME empresas...")
    df = pd.read_parquet(BORME_EMPRESAS)
    log.info("  Filas raw: %s", f"{len(df):,}")

    df["fecha_constitucion_parsed"] = parsear_fecha_constitucion(df["fecha_constitucion"])
    df["fecha_constitucion_parsed"] = pd.to_datetime(
        df["fecha_constitucion_parsed"], utc=True, errors="coerce"
    )
    df["fecha_borme"]  = pd.to_datetime(df["fecha_borme"], errors="coerce", utc=True)
    df["capital_euros"] = pd.to_numeric(df["capital_euros"], errors="coerce")
    df["_nombre_norm"]  = normalizar_nombre_empresa(df["empresa_norm"])
    df["_actos_lower"]  = df["actos"].astype(str).str.lower()

    # Constitución
    mask_const = df["_actos_lower"].str.contains("constituc", na=False)
    const = (
        df[mask_const].groupby("_nombre_norm")
        .agg(
            fecha_constitucion_min=("fecha_constitucion_parsed", "min"),
            fecha_borme_const     =("fecha_borme",               "min"),
        )
        .reset_index()
    )

    # Capital: último valor no-nulo por empresa
    capital = (
        df[df["capital_euros"].notna()]
        .sort_values("fecha_borme")
        .groupby("_nombre_norm")["capital_euros"]
        .last()
        .reset_index()
        .rename(columns={"capital_euros": "capital_ultimo"})
    )

    # Disolución
    mask_dis = df["_actos_lower"].str.contains(
        r"disoluci[oó]n|disuelt|liquidaci[oó]n|extinci[oó]n", na=False, regex=True
    )
    disol = (
        df[mask_dis].groupby("_nombre_norm")["fecha_borme"]
        .min().reset_index()
        .rename(columns={"fecha_borme": "fecha_disolucion"})
    )

    # Concurso
    mask_conc = df["_actos_lower"].str.contains(
        r"concurs|insolvenc|administraci[oó]n judicial", na=False, regex=True
    )
    conc = (
        df[mask_conc].groupby("_nombre_norm")["fecha_borme"]
        .min().reset_index()
        .rename(columns={"fecha_borme": "fecha_concurso"})
    )

    empresas = (
        const
        .merge(capital, on="_nombre_norm", how="outer")
        .merge(disol,   on="_nombre_norm", how="outer")
        .merge(conc,    on="_nombre_norm", how="outer")
    )

    # Fecha constitución final
    empresas["fecha_const_final"] = pd.to_datetime(
        empresas["fecha_constitucion_min"].fillna(empresas["fecha_borme_const"]),
        utc=True, errors="coerce"
    )
    empresas = empresas.drop(columns=["fecha_constitucion_min", "fecha_borme_const"])

    log.info("  Empresas únicas en BORME:     %s", f"{len(empresas):,}")
    log.info("  Con fecha constitución:       %s", f"{empresas['fecha_const_final'].notna().sum():,}")
    log.info("  Con capital conocido:         %s", f"{empresas['capital_ultimo'].notna().sum():,}")
    log.info("  Con acto de disolución:       %s", f"{empresas['fecha_disolucion'].notna().sum():,}")
    log.info("  Con acto de concurso:         %s", f"{empresas['fecha_concurso'].notna().sum():,}")

    return empresas


# ── PASO 4: Detectar colisiones ───────────────────────────────────────────────
def detectar_colisiones(tabla_nif: pd.DataFrame,
                         borme_empresas: pd.DataFrame) -> np.ndarray:
    conteo = (
        borme_empresas.groupby("_nombre_norm").size()
        .reset_index(name="n_borme")
    )
    merged = tabla_nif[["adjudicatario_nombre_norm"]].merge(
        conteo, left_on="adjudicatario_nombre_norm",
        right_on="_nombre_norm", how="left"
    )
    colision = (merged["n_borme"].fillna(0) > 1).values
    log.info("Colisiones de nombre: %s (%.1f%%)",
             f"{int(colision.sum()):,}",
             100 * colision.sum() / max(len(tabla_nif), 1))
    return colision


# ── FLAGS ─────────────────────────────────────────────────────────────────────
def _safe_days_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    """Calcula (a - b).dt.days de forma segura, manejando overflow."""
    try:
        return (a - b).dt.days
    except OverflowError:
        log.warning("  Overflow en resta de fechas, usando cálculo fila a fila.")
        def _safe(x, y):
            try:
                if pd.isna(x) or pd.isna(y):
                    return np.nan
                return (x - y).days
            except (OverflowError, ValueError):
                return np.nan
        return pd.Series(
            [_safe(x, y) for x, y in zip(a, b)],
            index=a.index,
        )

def calcular_F1(df: pd.DataFrame) -> pd.Series:
    dias = _safe_days_diff(df["fecha_primera_adj"], df["fecha_const_final"])
    f1 = (dias.notna() & (dias < UMBRAL_F1_DIAS)).astype(int)
    log.info("F1 (constitución <180d antes):   %s activos", f"{int(f1.sum()):,}")
    return f1

def calcular_F2(df: pd.DataFrame) -> pd.Series:
    f2 = (
        df["capital_ultimo"].notna()
        & (df["capital_ultimo"] < UMBRAL_F2_CAPITAL)
        & (df["importe_acumulado"] > UMBRAL_F2_IMPORTE)
    ).astype(int)
    log.info("F2 (capital bajo + importe alto): %s activos", f"{int(f2.sum()):,}")
    return f2

def calcular_F3(tabla_nif: pd.DataFrame,
                borme_cargos: pd.DataFrame,
                corpus: pd.DataFrame) -> pd.Series:
    log.info("Calculando F3 (administradores compartidos)...")
    if "persona_hash" not in borme_cargos.columns:
        log.warning("  F3: persona_hash no encontrada. F3 = 0.")
        return pd.Series(0, index=tabla_nif.index)

    borme_cargos = borme_cargos.copy()
    borme_cargos["_nombre_norm"] = normalizar_nombre_empresa(borme_cargos["empresa_norm"])
    borme_cargos["_tipo_lower"]  = borme_cargos["tipo_acto"].astype(str).str.lower()

    mask_nom = borme_cargos["_tipo_lower"].str.contains(
        r"nombramiento|administrador|consejero|apoderado", na=False
    )
    noms = (
        borme_cargos[mask_nom][["persona_hash", "_nombre_norm"]]
        .dropna().drop_duplicates()
    )

    pares = (
        corpus[corpus["organo_nif"].notna()]
        [["adjudicatario_nombre_norm", "organo_nif"]]
        .drop_duplicates()
        .rename(columns={"adjudicatario_nombre_norm": "_nombre_norm"})
    )

    red = noms.merge(pares, on="_nombre_norm", how="inner")
    conteo = (
        red.groupby(["persona_hash", "organo_nif"])["_nombre_norm"]
        .nunique().reset_index(name="n_empresas")
    )
    casos = conteo[conteo["n_empresas"] >= 2]
    nombres_f3 = set(
        red.merge(casos[["persona_hash", "organo_nif"]],
                  on=["persona_hash", "organo_nif"], how="inner")["_nombre_norm"]
    )

    f3 = tabla_nif["adjudicatario_nombre_norm"].isin(nombres_f3).astype(int)
    log.info("  F3 activos: %s  |  Pares (hash,órgano) ≥2 empresas: %s",
             f"{int(f3.sum()):,}", f"{len(casos):,}")
    return f3

def calcular_F4(df: pd.DataFrame) -> pd.Series:
    dias = _safe_days_diff(df["fecha_disolucion"], df["fecha_ultima_adj"])
    f4 = (dias.notna() & (dias >= 0) & (dias <= UMBRAL_F4_DIAS)).astype(int)
    log.info("F4 (disolución <12m post-adj):   %s activos", f"{int(f4.sum()):,}")
    return f4

def calcular_F5(df: pd.DataFrame) -> pd.Series:
    try:
        f5 = (
            df["fecha_concurso"].notna()
            & (df["fecha_concurso"] <= df["fecha_ultima_adj"])
        ).astype(int)
    except OverflowError:
        log.warning("  F5: overflow en comparación de fechas, usando cálculo seguro.")
        dias = _safe_days_diff(df["fecha_concurso"], df["fecha_ultima_adj"])
        f5 = (dias.notna() & (dias <= 0)).astype(int)
    log.info("F5 (empresa en concurso):        %s activos", f"{int(f5.sum()):,}")
    return f5


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("SCRIPT 01 — Flags corporativos BORME × Contratación 2019-2025")
    log.info("Inicio: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    corpus     = cargar_corpus()
    tabla_nif  = construir_tabla_nif(corpus)
    borme_emp  = preparar_borme_empresas()
    borme_carg = pd.read_parquet(BORME_CARGOS)

    tabla_nif["join_colision_nombre"] = detectar_colisiones(tabla_nif, borme_emp)

    # Join corpus ↔ BORME por nombre_norm
    merged = tabla_nif.merge(
        borme_emp,
        left_on="adjudicatario_nombre_norm",
        right_on="_nombre_norm",
        how="left",
    ).drop(columns=["_nombre_norm"], errors="ignore")

    n_match = merged["fecha_const_final"].notna().sum()
    log.info("Match en BORME: %s / %s (%.1f%%)",
             f"{n_match:,}", f"{len(merged):,}",
             100 * n_match / max(len(merged), 1))

    merged["F1"] = calcular_F1(merged)
    merged["F2"] = calcular_F2(merged)
    merged["F3"] = calcular_F3(merged, borme_carg, corpus)
    merged["F4"] = calcular_F4(merged)
    merged["F5"] = calcular_F5(merged)

    merged["score_corporativo"]    = merged[["F1","F2","F3","F4","F5"]].sum(axis=1)
    merged["borme_cobertura"]      = merged["fecha_const_final"].notna()
    merged["flags_fiabilidad"]     = merged["join_colision_nombre"].map(
        {True: "indeterminado", False: "fiable"}
    )

    cols_salida = [
        "adjudicatario_nif", "adjudicatario_nombre", "adjudicatario_nombre_norm",
        "fecha_primera_adj", "fecha_ultima_adj",
        "importe_acumulado", "n_contratos",
        "fecha_const_final", "capital_ultimo",
        "fecha_disolucion", "fecha_concurso",
        "F1", "F2", "F3", "F4", "F5",
        "score_corporativo",
        "borme_cobertura", "join_colision_nombre", "flags_fiabilidad",
    ]
    resultado = merged[[c for c in cols_salida if c in merged.columns]].copy()

    out_path = OUT_DIR / "flags_corporativos.parquet"
    resultado.to_parquet(out_path, index=False)
    log.info("Guardado: %s  (%s filas)", out_path, f"{len(resultado):,}")

    # ── Quality report ────────────────────────────────────────────────────────
    n  = len(resultado)
    nc = int(resultado["borme_cobertura"].sum())
    ni = int(resultado["join_colision_nombre"].sum())
    sub = resultado[resultado["borme_cobertura"]]

    qpath = REPORTS_DIR / "flags_corporativos_quality.txt"
    with open(qpath, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("INFORME DE CALIDAD — FLAGS CORPORATIVOS (Script 01 v2)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 65 + "\n\n")

        f.write("--- Cobertura del join ---\n")
        f.write(f"  Adjudicatarios únicos en corpus:        {n:>10,}\n")
        f.write(f"  Con match en BORME:                     {nc:>10,}  ({100*nc/n:.1f}%)\n")
        f.write(f"  Sin match en BORME:                     {n-nc:>10,}  ({100*(n-nc)/n:.1f}%)\n")
        f.write(f"  Con colisión de nombre (indeterminado): {ni:>10,}  ({100*ni/n:.1f}%)\n\n")

        f.write("--- Activación por flag (sobre total corpus) ---\n")
        for flag in ["F1","F2","F3","F4","F5"]:
            nf = int(resultado[flag].sum())
            f.write(f"  {flag}: {nf:>8,}  ({100*nf/n:.2f}%)\n")

        f.write("\n--- Activación por flag (solo NIFs con cobertura BORME) ---\n")
        for flag in ["F1","F2","F3","F4","F5"]:
            nf = int(sub[flag].sum())
            f.write(f"  {flag}: {nf:>8,}  ({100*nf/max(len(sub),1):.2f}%)\n")

        f.write("\n--- Distribución score corporativo (0-5) ---\n")
        for sc, cnt in resultado["score_corporativo"].value_counts().sort_index().items():
            f.write(f"  Score {int(sc)}: {int(cnt):>8,}  ({100*cnt/n:.1f}%)\n")

        f.write("\n--- Advertencias metodológicas ---\n")
        f.write("  Join realizado por nombre normalizado (BORME no tiene NIF).\n")
        f.write("  Colisiones = nombre_norm de un adjudicatario coincide con\n")
        f.write("  >1 empresa en BORME. Sus flags se marcan 'indeterminado'.\n")
        f.write("  No usar flags 'indeterminado' en análisis cuantitativo\n")
        f.write("  sin verificación manual previa.\n")
        f.write(f"\nOutput: {out_path}\n")

    log.info("Quality report: %s", qpath)
    log.info("=" * 65)
    log.info("Script 01 completado.")
    log.info("SIGUIENTE: script_02_calcular_indicadores_comportamiento.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
