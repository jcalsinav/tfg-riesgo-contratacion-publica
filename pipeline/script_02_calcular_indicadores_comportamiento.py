"""
script_02_calcular_indicadores_comportamiento.py  (v2 — post-diagnóstico)
=========================================================================
Calcula los seis indicadores de comportamiento en contratación
(B1–B6) sobre el corpus curado 2019-2025.

Hallazgos del diagnóstico que condicionan el diseño
----------------------------------------------------
- procedimiento_code es dtype str/object → comparar con strings, no ints
- Mapa de códigos confirmado:
    '1'   Abierto
    '2'   Restringido
    '3'   Negociado con publicidad
    '4'   Negociado sin publicidad  ← B6
    '6'   Asociación innovación (valor basura/default de PLACSP)
    '7'   Contrato menor            ← equivalente a es_menor==True
    '9'   Basado en sistema dinámico
    '100' Basado en acuerdo marco
    '999' Otros
    '13'  Sin decodificar
- procedimiento_code=='7' ↔ es_menor==True son la misma población (112K)
- procedimiento_code=='6' es un comodín: mediana importe 1.134€,
  muchos son menores encubiertos → Opción D híbrida para clasificarlos

Clasificación es_menor_v2 (Opción D — usada en B4)
---------------------------------------------------
  Tier 1 (declarado):   es_menor == True
  Tier 2 (encubierto):  procedimiento_code == '6'
                        AND importe < 15.000€ (servicios/suministros)
                        AND importe < 40.000€ (obras)

Indicadores calculados
----------------------
  B1  Proporción de adjudicaciones con oferta única
      → Solo sobre procedimientos '1','2','3' con num_ofertas válido
  B2  HHI por órgano (concentración de gasto por órgano-año)
  B3  Racha máxima de adjudicaciones consecutivas (órgano, cpv2d)
  B4  Clustering bajo umbral contrato menor
      → Usa es_menor_v2 (Opción D)
  B5  Clustering bajo umbral SARA (10% inferior al umbral por bienio)
      → Solo sobre contratos NO menores
  B6  Tasa de procedimientos excepcionales en no-menores
      → procedimiento_code == '4' (negociado sin publicidad)

Outputs
-------
  Nacional/curated/indicadores_comportamiento_raw.parquet
  Nacional/curated/indicadores_comportamiento_flags.parquet
  Nacional/reports/indicadores_comportamiento_quality.txt
  Nacional/logs/script_02.log

Prerequisito
------------
  Scripts de preparación por CCAA ya ejecutados.
  Script 01 NO es prerequisito directo (no consume su output).
"""

from __future__ import annotations
from pathlib import Path
import logging
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd

# ── RUTAS ─────────────────────────────────────────────────────────────────────
BASE = Path("/Volumes/Datos Jan/Base De Datos TFG JAN/Base de Datos")

CURADOS = {
    "placsp":    BASE / "Nacional/PLACSP/curated/placsp_2019_2025_curated.parquet",
    "andalucia": BASE / "CCAA/Andalucia/curated/andalucia_2019_2025_curated.parquet",
    "cataluna":  BASE / "CCAA/Cataluna/curated/cataluna_2019_2025_contratos_registro.parquet",
    "cat_men":   BASE / "CCAA/Cataluna/curated/cataluna_2019_2025_menores.parquet",
    "madrid":    BASE / "CCAA/Comunidad_Madrid/curated/madrid_cam_2019_2025_curated.parquet",
    "euskadi":   BASE / "CCAA/Euskadi/curated/euskadi_2019_2025_curated.parquet",
    "galicia":   BASE / "CCAA/Galicia/curated/galicia_2019_2025_curated.parquet",
    "valencia":  BASE / "CCAA/Valencia/curated/valencia_2019_2025_regcon_consolidado.parquet",
}

# Fallback CCAA cuando la fuente no tiene campo NUTS
FUENTE_A_CCAA = {
    "placsp": "NAC", "andalucia": "AN", "cataluna": "CT",
    "cat_men": "CT", "madrid": "MD", "euskadi": "PV",
    "galicia": "GA", "valencia": "VC",
}

OUT_DIR     = BASE / "Nacional/curated"
REPORTS_DIR = BASE / "Nacional/reports"
LOGS_DIR    = BASE / "Nacional/logs"
for d in [OUT_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── LOGGING ───────────────────────────────────────────────────────────────────
log_path = LOGS_DIR / "script_02.log"
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

# Códigos de procedimiento con competencia real → usados en B1
PROC_COMPETENCIA = {"1", "2", "3"}           # abierto, restringido, neg c/ pub

# Código de procedimiento excepcional → usado en B6
PROC_EXCEPCIONAL = {"4"}                     # negociado sin publicidad

# Código basura/default de PLACSP → Tier 2 en es_menor_v2
PROC_BASURA = "6"

# Umbrales contrato menor (art. 118 LCSP)
UMBRAL_MENOR_SERVICIOS = 15_000
UMBRAL_MENOR_OBRAS     = 40_000

# Umbrales SARA por bienio {año_inicio: {tipo: umbral}}
UMBRALES_SARA = {
    2016: {"obras": 5_225_000, "servicios": 209_000},
    2018: {"obras": 5_548_000, "servicios": 221_000},
    2020: {"obras": 5_350_000, "servicios": 214_000},
    2022: {"obras": 5_382_000, "servicios": 215_000},
    2024: {"obras": 5_538_000, "servicios": 221_000},
}
BANDA_SARA_PCT = 0.10    # 10% inferior al umbral SARA

# Percentil para umbral de alerta binario (metodología 4.c)
P_ALERTA = 0.75

# Umbral absoluto para clustering (B4 y B5)
UMBRAL_CLUSTERING_ABS = 0.20   # > 20% de contratos en la banda


# ── UTILIDADES ────────────────────────────────────────────────────────────────
def normalizar_nif(serie: pd.Series) -> pd.Series:
    return (
        serie.astype(str).str.upper()
        .str.replace(r"[\s\-\.]", "", regex=True).str.strip()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan, "NULL": np.nan})
    )


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


def normalizar_nombre(serie: pd.Series) -> pd.Series:
    FORMAS = re.compile(
        r"\b(s\.?l\.?u?\.?|s\.?a\.?u?\.?|s\.?l\.?l\.?|coop\.?|"
        r"sociedad limitada|sociedad anonima|sl|sa|sau|slu)\b",
        re.IGNORECASE,
    )
    s = serie.astype(str).str.lower().str.strip()
    s = s.apply(lambda x: unicodedata.normalize("NFKD", x)
                .encode("ascii", "ignore").decode("ascii"))
    s = s.apply(lambda x: FORMAS.sub("", x))
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.replace({"nan": np.nan, "": np.nan})


def extraer_cpv2d(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.strip().str.extract(
        r"^(\d{2})", expand=False
    ).fillna("XX")


def nuts_a_ccaa(nuts: pd.Series, fallback: str) -> pd.Series:
    MAPA = {
        "ES11":"GA","ES12":"AS","ES13":"CB","ES21":"PV","ES22":"NA",
        "ES23":"RI","ES24":"AR","ES30":"MD","ES41":"CL","ES42":"CM",
        "ES43":"EX","ES51":"CT","ES52":"VC","ES53":"IB","ES61":"AN",
        "ES62":"MU","ES63":"CE","ES64":"ML","ES70":"CN",
    }
    return nuts.astype(str).str[:4].str.upper().map(MAPA).fillna(fallback)


def get_umbral_sara(anio: int, tipo_contrato: str) -> float:
    bienio = max(b for b in UMBRALES_SARA if b <= anio)
    es_obra = "obra" in str(tipo_contrato).lower()
    return UMBRALES_SARA[bienio]["obras" if es_obra else "servicios"]


# ── CLASIFICACIÓN es_menor_v2 (Opción D híbrida) ─────────────────────────────
def clasificar_es_menor_v2(df: pd.DataFrame) -> pd.Series:
    """
    Tier 1: es_menor == True
    Tier 2: procedimiento_code == '6' AND importe < umbral según tipo contrato
    """
    es_obra = df["tipo_contrato"].astype(str).str.lower().str.contains(
        "obra", na=False
    )
    umbral = np.where(es_obra, UMBRAL_MENOR_OBRAS, UMBRAL_MENOR_SERVICIOS)

    tiene_proc_code = "procedimiento_code" in df.columns
    if tiene_proc_code:
        tier2 = (
            (df["procedimiento_code"].astype(str).str.strip() == PROC_BASURA)
            & (pd.to_numeric(df["importe"], errors="coerce") < umbral)
        )
    else:
        tier2 = pd.Series(False, index=df.index)

    return df["es_menor"].fillna(False) | tier2.fillna(False)


# ── PASO 1: Cargar corpus unificado ──────────────────────────────────────────
def cargar_corpus() -> pd.DataFrame:
    """
    Carga todos los curados y construye la tabla analítica unificada con:
    adjudicatario_key, organo_key, ccaa, cpv2d, importe,
    num_ofertas, es_menor, es_menor_v2, procedimiento_code,
    tipo_contrato, fecha, anio, fuente
    """
    partes = []

    for fuente, ruta in CURADOS.items():
        if not ruta.exists():
            log.warning("Curado no encontrado, se omite: %s", ruta)
            continue
        log.info("Cargando: %s", fuente)
        df = pd.read_parquet(ruta)
        tmp = pd.DataFrame(index=df.index)

        # ── Clave adjudicatario ──────────────────────────────────────────────
        col_nif = _find_col(df, [
            "adjudicatario_nif", "nif_adjudicatario", "nif",
            "nif_adj", "adjudicatario nif", "nif del adjudicatario",
            "nif_empresa", "empresa_nif", "cif", "adjudicatario_cif",
            "cif_nif_enmascarado",                          # Valencia REGCON
        ])
        col_nom = _find_col(df, [
            "adjudicatario_nombre", "adjudicatario", "razon_social_adjudicatario",
            "nombre_adjudicatario", "empresa", "razon_social", "nombre_empresa",
            "adjudicatario nombre", "nombre del adjudicatario",
            "nombre_o_razon_social",                        # Valencia REGCON
        ])
        nif  = normalizar_nif(df[col_nif]) if col_nif else pd.Series(np.nan, index=df.index)
        nom  = normalizar_nombre(df[col_nom]) if col_nom else pd.Series(np.nan, index=df.index)
        tmp["adjudicatario_key"] = nif.fillna(nom)

        # ── Clave órgano ─────────────────────────────────────────────────────
        col_org_nif = _find_col(df, [
            "organo_nif", "nif_organo", "nif_organo_contratante",
            "organo_contratante_nif", "nif del organo",
        ])
        col_org_nom = _find_col(df, [
            "organo_nombre", "organo_contratante", "entidad_adjudicadora",
            "Entidad Adjudicadora", "poder_adjudicador", "perfil_contratante",
            "conselleria_ent_adjud",                        # Valencia REGCON
        ])
        org_nif = normalizar_nif(df[col_org_nif]) if col_org_nif else pd.Series(np.nan, index=df.index)
        org_nom = normalizar_nombre(df[col_org_nom]) if col_org_nom else pd.Series(np.nan, index=df.index)
        tmp["organo_key"] = org_nif.fillna(org_nom)

        # ── CCAA ─────────────────────────────────────────────────────────────
        col_nuts = _find_col(df, ["nuts", "nuts_code", "ubicacion"])
        if col_nuts and fuente == "placsp":
            tmp["ccaa"] = nuts_a_ccaa(df[col_nuts], "NAC")
        else:
            tmp["ccaa"] = FUENTE_A_CCAA.get(fuente, "NAC")

        # ── CPV 2 dígitos ─────────────────────────────────────────────────────
        col_cpv = _find_col(df, [
            "cpv_principal", "cpv_code", "codigo_cpv", "cpv",
            "cpv2d", "codigo_cpv_principal",
            "codigo_cpv",                                   # Valencia REGCON (alias)
        ])
        tmp["cpv2d"] = extraer_cpv2d(df[col_cpv]) if col_cpv else "XX"

        # ── Importe ──────────────────────────────────────────────────────────
        col_imp = _find_col(df, [
            "importe_adjudicacion", "importe_licitacion", "importe",
            "importe de adjudicacion", "importe_adj", "precio_adjudicacion",
            "importe adjudicacion", "valor_adjudicacion",
            "imp_adjud_sin_iva",                            # Valencia REGCON (preferido)
            "imp_total_adjud",                             # Valencia REGCON (alternativo)
            "importe_licit_sin_iva",                       # Valencia REGCON (licitación)
        ])
        tmp["importe"] = _parse_importe(df[col_imp]) if col_imp else np.nan

        # ── num_ofertas ───────────────────────────────────────────────────────
        col_of = _find_col(df, [
            "num_ofertas", "numero_ofertas", "n_ofertas",
            "nº de ofertas", "num_licitadores",             # Valencia REGCON
            "num_empresas_invitadas",                       # Valencia REGCON (alternativo)
        ])
        tmp["num_ofertas"] = pd.to_numeric(df[col_of], errors="coerce") if col_of else np.nan

        # ── es_menor (Tier 1) ─────────────────────────────────────────────────
        if "es_menor" in df.columns:
            tmp["es_menor"] = df["es_menor"].astype(bool)
        elif "es_menor_explicito" in df.columns:
            tmp["es_menor"] = df["es_menor_explicito"].astype(bool)
        else:
            tmp["es_menor"] = False

        # ── procedimiento_code ────────────────────────────────────────────────
        col_proc = _find_col(df, [
            "procedimiento_code", "procedimiento",
            "procedimiento",                                # Valencia REGCON (mismo nombre)
        ])
        if col_proc:
            p_s = df[col_proc].astype(str).str.strip()
            mask_abierto = p_s.str.contains("abierto", case=False, na=False)
            mask_restringido = p_s.str.contains("restringido", case=False, na=False)
            mask_neg_sin = p_s.str.contains("sin publicidad", case=False, na=False) & p_s.str.contains("negociado", case=False, na=False)
            mask_neg_con = p_s.str.contains("negociado", case=False, na=False) & ~mask_neg_sin
            p_s = p_s.mask(mask_abierto, "1").mask(mask_restringido, "2").mask(mask_neg_con, "3").mask(mask_neg_sin, "4")
            tmp["procedimiento_code"] = p_s
        else:
            tmp["procedimiento_code"] = np.nan

        # ── tipo_contrato ─────────────────────────────────────────────────────
        col_tipo = _find_col(df, [
            "tipo_contrato", "tipo_de_contrato", "Tipo de contrato",
            "clase_de_contrato",                            # Valencia REGCON
            "tipo",                                         # Valencia REGCON (campo TIPO)
        ])
        tmp["tipo_contrato"] = df[col_tipo].astype(str) if col_tipo else ""

        # ── Fecha y año ───────────────────────────────────────────────────────
        col_fecha = _find_col(df, [
            "fecha_referencia", "fecha_adjudicacion", "fecha_publicacion",
            "fecha_formalizacion",                          # Valencia REGCON (preferido)
            "fecha_publ_adj_form_docv",                    # Valencia REGCON (alternativo)
            "fecha_publ_licit_perfil",                     # Valencia REGCON (licitación)
        ])
        tmp["fecha"] = pd.to_datetime(
            df[col_fecha], errors="coerce", utc=True
        ) if col_fecha else pd.NaT
        tmp["anio"] = tmp["fecha"].dt.year

        tmp["fuente"] = fuente
        partes.append(tmp)

    corpus = pd.concat(partes, ignore_index=True)

    # Descartar sin clave de adjudicatario
    sin_adj = corpus["adjudicatario_key"].isna()
    n_total = len(corpus)
    corpus  = corpus[~sin_adj].copy()

    # es_menor_v2 (Opción D híbrida)
    corpus["es_menor_v2"] = clasificar_es_menor_v2(corpus)

    # Resumen
    log.info("Corpus: %s filas | Descartadas (sin adj key): %s",
             f"{n_total:,}", f"{int(sin_adj.sum()):,}")
    log.info("es_menor Tier1: %s | es_menor_v2 (D): %s",
             f"{corpus['es_menor'].sum():,}", f"{corpus['es_menor_v2'].sum():,}")
    log.info("Con num_ofertas: %s (%.1f%%) | Con organo_key: %s (%.1f%%)",
             f"{corpus['num_ofertas'].notna().sum():,}",
             100 * corpus["num_ofertas"].notna().mean(),
             f"{corpus['organo_key'].notna().sum():,}",
             100 * corpus["organo_key"].notna().mean())

    return corpus


# ── B1: Oferta única ──────────────────────────────────────────────────────────
def calcular_B1(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Solo sobre procedimientos con competencia real (códigos '1','2','3')
    y num_ofertas >= 1.
    B1 = % de contratos con num_ofertas == 1.
    """
    log.info("Calculando B1 (oferta única)...")

    sub = corpus[
        corpus["procedimiento_code"].isin(PROC_COMPETENCIA) &
        corpus["num_ofertas"].notna() &
        (corpus["num_ofertas"] >= 1)
    ].copy()

    log.info("  Registros válidos para B1: %s", f"{len(sub):,}")
    if len(sub) == 0:
        log.warning("  B1: Sin registros válidos.")
        return pd.DataFrame(columns=["adjudicatario_key","ccaa","cpv2d",
                                     "B1_ratio","B1_n"])

    sub["_unica"] = (sub["num_ofertas"] == 1).astype(int)
    b1 = (
        sub.groupby(["adjudicatario_key","ccaa","cpv2d"])
        .agg(B1_ratio=("_unica","mean"), B1_n=("num_ofertas","count"))
        .reset_index()
    )
    log.info("  Grupos con B1: %s | Media B1_ratio: %.3f",
             f"{len(b1):,}", b1["B1_ratio"].mean())
    return b1


# ── B2: HHI por órgano ───────────────────────────────────────────────────────
def calcular_B2(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    HHI por órgano-año, luego asignado al adjudicatario como promedio
    ponderado por importe de los órganos en que operó.
    """
    log.info("Calculando B2 (HHI por órgano)...")

    sub = corpus[
        corpus["importe"].notna() &
        corpus["organo_key"].notna() &
        (corpus["importe"] > 0)
    ].copy()

    if len(sub) == 0:
        log.warning("  B2: Sin registros válidos.")
        return pd.DataFrame(columns=["adjudicatario_key","ccaa","cpv2d","B2_hhi_medio"])

    # Gasto total por (órgano, ccaa, anio)
    total_org = (
        sub.groupby(["organo_key","ccaa","anio"])["importe"]
        .sum().reset_index().rename(columns={"importe":"_total"})
    )
    # Gasto por (órgano, adjudicatario, ccaa, anio)
    por_adj = (
        sub.groupby(["organo_key","adjudicatario_key","ccaa","anio"])["importe"]
        .sum().reset_index().rename(columns={"importe":"_imp_adj"})
    )

    hhi_base = por_adj.merge(total_org, on=["organo_key","ccaa","anio"], how="left")
    hhi_base["_share_sq"] = (hhi_base["_imp_adj"] / hhi_base["_total"]) ** 2

    hhi_org = (
        hhi_base.groupby(["organo_key","ccaa","anio"])["_share_sq"]
        .sum().reset_index().rename(columns={"_share_sq":"hhi"})
    )
    hhi_base = hhi_base.merge(hhi_org, on=["organo_key","ccaa","anio"], how="left")

    # CPV modal por (adjudicatario, órgano, ccaa)
    cpv_modal = (
        sub.groupby(["adjudicatario_key","organo_key","ccaa"])["cpv2d"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "XX")
        .reset_index()
    )
    hhi_base = hhi_base.merge(cpv_modal,
                               on=["adjudicatario_key","organo_key","ccaa"],
                               how="left")
    hhi_base["cpv2d"] = hhi_base["cpv2d"].fillna("XX")

    # HHI medio ponderado por importe del adjudicatario en ese órgano
    def _hhi_ponderado(g):
        w = g["_imp_adj"].clip(lower=0)
        return np.average(g["hhi"], weights=w) if w.sum() > 0 else g["hhi"].mean()

    b2 = (
        hhi_base.groupby(["adjudicatario_key","ccaa","cpv2d"])
        .apply(_hhi_ponderado)
        .reset_index(name="B2_hhi_medio")
    )
    log.info("  Grupos con B2: %s | Media HHI: %.3f",
             f"{len(b2):,}", b2["B2_hhi_medio"].mean())
    return b2


# ── B3: Racha máxima de adjudicaciones consecutivas ──────────────────────────
def calcular_B3(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada (órgano, cpv2d), ordena contratos por fecha y calcula
    la longitud de racha ininterrumpida del mismo adjudicatario.
    B3 = máxima racha por (adjudicatario, ccaa, cpv2d).
    """
    log.info("Calculando B3 (racha consecutiva)...")

    sub = corpus[
        corpus["fecha"].notna() &
        corpus["organo_key"].notna()
    ].sort_values(["organo_key","cpv2d","fecha"]).reset_index(drop=True)

    if len(sub) == 0:
        log.warning("  B3: Sin registros válidos.")
        return pd.DataFrame(columns=["adjudicatario_key","ccaa","cpv2d","B3_racha_max"])

    n_grupos = sub.groupby(["organo_key","cpv2d"]).ngroups
    log.info("  Procesando %s grupos (organo, cpv2d)...", f"{n_grupos:,}")

    # Calcular rachas con vectorización parcial (más rápido que apply fila a fila)
    # 1. Identificar cambio de adjudicatario dentro del mismo grupo
    sub["_grupo"] = sub.groupby(["organo_key","cpv2d"]).ngroup()
    sub["_cambio"] = (
        (sub["adjudicatario_key"] != sub["adjudicatario_key"].shift(1)) |
        (sub["_grupo"] != sub["_grupo"].shift(1))
    ).astype(int)
    # 2. ID de racha acumulado
    sub["_racha_id"] = sub["_cambio"].cumsum()
    # 3. Tamaño de cada racha
    racha_size = (
        sub.groupby("_racha_id")["adjudicatario_key"]
        .transform("count")
    )
    sub["_racha_len"] = racha_size

    b3 = (
        sub.groupby(["adjudicatario_key","ccaa","cpv2d"])["_racha_len"]
        .max().reset_index().rename(columns={"_racha_len":"B3_racha_max"})
    )
    log.info("  Grupos con B3: %s | Media: %.2f | Máx: %s",
             f"{len(b3):,}", b3["B3_racha_max"].mean(),
             int(b3["B3_racha_max"].max()))
    return b3


# ── B4: Clustering bajo umbral contrato menor ────────────────────────────────
def calcular_B4(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Usa es_menor_v2 (Opción D híbrida).
    Banda: [umbral * 0.90, umbral] según tipo de contrato.
    B4 = % de contratos menores del adjudicatario en esa banda.
    """
    log.info("Calculando B4 (clustering umbral menor)...")

    menores = corpus[corpus["es_menor_v2"] & corpus["importe"].notna()].copy()
    log.info("  Contratos menores (es_menor_v2): %s", f"{len(menores):,}")

    if len(menores) == 0:
        log.warning("  B4: Sin contratos menores.")
        return pd.DataFrame(columns=["adjudicatario_key","ccaa","cpv2d",
                                     "B4_ratio","B4_n"])

    es_obra = menores["tipo_contrato"].str.lower().str.contains("obra", na=False)
    menores["_umbral"]    = np.where(es_obra, UMBRAL_MENOR_OBRAS, UMBRAL_MENOR_SERVICIOS)
    menores["_banda_inf"] = menores["_umbral"] * 0.90
    menores["_en_banda"]  = (
        (menores["importe"] >= menores["_banda_inf"]) &
        (menores["importe"] <= menores["_umbral"])
    ).astype(int)

    b4 = (
        menores.groupby(["adjudicatario_key","ccaa","cpv2d"])
        .agg(B4_ratio=("_en_banda","mean"), B4_n=("importe","count"))
        .reset_index()
    )
    log.info("  Grupos con B4: %s | Media B4_ratio: %.3f",
             f"{len(b4):,}", b4["B4_ratio"].mean())
    return b4


# ── B5: Clustering bajo umbral SARA ──────────────────────────────────────────
def calcular_B5(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Solo sobre contratos NO menores con fecha e importe válidos.
    Banda: [umbral_sara * 0.90, umbral_sara) según bienio y tipo.
    B5 = % de contratos no-menores del adjudicatario en esa banda.
    """
    log.info("Calculando B5 (clustering umbral SARA)...")

    sub = corpus[
        ~corpus["es_menor_v2"] &
        corpus["importe"].notna() &
        corpus["anio"].notna()
    ].copy()
    sub["anio"] = sub["anio"].astype(int)

    if len(sub) == 0:
        log.warning("  B5: Sin contratos no-menores válidos.")
        return pd.DataFrame(columns=["adjudicatario_key","ccaa","cpv2d",
                                     "B5_ratio","B5_n"])

    # Umbral SARA vectorizado por bienio
    # (más rápido que apply fila a fila para millones de filas)
    sub["_umbral_sara"] = sub.apply(
        lambda r: get_umbral_sara(r["anio"], r["tipo_contrato"]), axis=1
    )
    sub["_banda_inf"]   = sub["_umbral_sara"] * (1 - BANDA_SARA_PCT)
    sub["_en_banda"]    = (
        (sub["importe"] >= sub["_banda_inf"]) &
        (sub["importe"] <  sub["_umbral_sara"])
    ).astype(int)

    b5 = (
        sub.groupby(["adjudicatario_key","ccaa","cpv2d"])
        .agg(B5_ratio=("_en_banda","mean"), B5_n=("importe","count"))
        .reset_index()
    )
    log.info("  Grupos con B5: %s | Media B5_ratio: %.3f",
             f"{len(b5):,}", b5["B5_ratio"].mean())
    return b5


# ── B6: Procedimientos excepcionales en no-menores ───────────────────────────
def calcular_B6(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Solo sobre contratos NO menores.
    Usa procedimiento_code directamente: código '4' = negociado sin publicidad.
    B6 = % de no-menores del adjudicatario adjudicados con código '4'.
    """
    log.info("Calculando B6 (procedimientos excepcionales)...")

    sub = corpus[~corpus["es_menor_v2"]].copy()

    if "procedimiento_code" not in sub.columns or sub["procedimiento_code"].isna().all():
        log.warning("  B6: procedimiento_code no disponible.")
        return pd.DataFrame(columns=["adjudicatario_key","ccaa","cpv2d",
                                     "B6_ratio","B6_n"])

    sub["_excepcional"] = (
        sub["procedimiento_code"].astype(str).str.strip().isin(PROC_EXCEPCIONAL)
    ).astype(int)

    b6 = (
        sub.groupby(["adjudicatario_key","ccaa","cpv2d"])
        .agg(B6_ratio=("_excepcional","mean"), B6_n=("_excepcional","count"))
        .reset_index()
    )
    log.info("  Grupos con B6: %s | Media B6_ratio: %.3f",
             f"{len(b6):,}", b6["B6_ratio"].mean())
    return b6


# ── COMBINAR ──────────────────────────────────────────────────────────────────
def combinar(corpus, b1, b2, b3, b4, b5, b6) -> pd.DataFrame:
    base = corpus[["adjudicatario_key","ccaa","cpv2d"]].drop_duplicates().copy()
    for df_ind in [b1, b2, b3, b4, b5, b6]:
        if df_ind is not None and len(df_ind) > 0:
            base = base.merge(
                df_ind, on=["adjudicatario_key","ccaa","cpv2d"], how="left"
            )
    # Métricas de volumen
    vol = corpus.groupby(["adjudicatario_key","ccaa","cpv2d"]).agg(
        n_contratos  =("adjudicatario_key","count"),
        importe_total=("importe","sum"),
        fecha_min    =("fecha","min"),
        fecha_max    =("fecha","max"),
    ).reset_index()
    base = base.merge(vol, on=["adjudicatario_key","ccaa","cpv2d"], how="left")
    log.info("Tabla raw: %s filas × %s cols", f"{len(base):,}", len(base.columns))
    return base


# ── BINARIZAR (p75 y absoluto) ────────────────────────────────────────────────

# Umbral fijo para flag_B1
# B1_ratio es muy bimodal (0 o 1): el p75 dentro de (ccaa, cpv2d) es = 1.0
# en la mayoría de grupos, haciendo que la condición > p75 sea imposible
# (nada puede superar 1.0). Esto produce 0 flags en CCAA como Valencia.
# Fix: umbral fijo B1_ratio > 0.5 con mínimo de observaciones B1_n >= 3.
# Interpretación: más de la mitad de las licitaciones competitivas del grupo
# resultaron en oferta única, con al menos 3 licitaciones observadas.
UMBRAL_B1_RATIO = 0.50
UMBRAL_B1_N_MIN = 3


def binarizar(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Genera flag_B1..flag_B6 y score_comportamiento.
    B1             → umbral fijo: B1_ratio > 0.50 AND B1_n >= 3
    B2, B3, B6     → umbral p75 dentro de (ccaa, cpv2d)
    B4, B5         → umbral absoluto 0.20
    """
    flags = raw[["adjudicatario_key","ccaa","cpv2d",
                 "n_contratos","importe_total"]].copy()

    # ── flag_B1: umbral fijo (no p75) ────────────────────────────────────────
    if "B1_ratio" not in raw.columns or "B1_n" not in raw.columns:
        flags["flag_B1"] = np.nan
    else:
        flags["flag_B1"] = (
            (raw["B1_ratio"] > UMBRAL_B1_RATIO) &
            (raw["B1_n"]    >= UMBRAL_B1_N_MIN)
        ).astype(float)
        flags.loc[raw["B1_ratio"].isna(), "flag_B1"] = np.nan
        n_act = int(flags["flag_B1"].sum())
        log.info("  B1_ratio → flag_B1 (umbral >%.2f AND B1_n>=%d): %s activos (%.1f%%)",
                 UMBRAL_B1_RATIO, UMBRAL_B1_N_MIN,
                 f"{n_act:,}", 100 * n_act / max(len(flags), 1))

    # ── p75 para B2, B3, B6 ──────────────────────────────────────────────────
    for col, fcol in [
        ("B2_hhi_medio","flag_B2"),
        ("B3_racha_max","flag_B3"),
        ("B6_ratio",    "flag_B6"),
    ]:
        if col not in raw.columns:
            flags[fcol] = np.nan
            continue
        p75 = raw.groupby(["ccaa","cpv2d"])[col].transform(
            lambda x: x.quantile(P_ALERTA)
        )
        flags[fcol] = (raw[col] > p75).astype(float)
        flags.loc[raw[col].isna(), fcol] = np.nan
        n_act = int(flags[fcol].sum())
        log.info("  %s → %s: %s activos (%.1f%%)",
                 col, fcol, f"{n_act:,}",
                 100 * n_act / max(len(flags), 1))

    # absoluto
    for col, fcol in [("B4_ratio","flag_B4"), ("B5_ratio","flag_B5")]:
        if col not in raw.columns:
            flags[fcol] = np.nan
            continue
        flags[fcol] = (raw[col] > UMBRAL_CLUSTERING_ABS).astype(float)
        flags.loc[raw[col].isna(), fcol] = np.nan
        n_act = int(flags[fcol].sum())
        log.info("  %s → %s (umbral %.2f): %s activos (%.1f%%)",
                 col, fcol, UMBRAL_CLUSTERING_ABS,
                 f"{n_act:,}", 100 * n_act / max(len(flags), 1))

    flag_cols = ["flag_B1","flag_B2","flag_B3","flag_B4","flag_B5","flag_B6"]
    flags["score_comportamiento"] = flags[flag_cols].sum(axis=1, min_count=1)
    return flags


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info("SCRIPT 02 — Indicadores de comportamiento 2019-2025  (v2)")
    log.info("Inicio: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    corpus = cargar_corpus()

    b1 = calcular_B1(corpus)
    b2 = calcular_B2(corpus)
    b3 = calcular_B3(corpus)
    b4 = calcular_B4(corpus)
    b5 = calcular_B5(corpus)
    b6 = calcular_B6(corpus)

    raw   = combinar(corpus, b1, b2, b3, b4, b5, b6)
    flags = binarizar(raw)

    raw_path   = OUT_DIR / "indicadores_comportamiento_raw.parquet"
    flags_path = OUT_DIR / "indicadores_comportamiento_flags.parquet"
    raw.to_parquet(raw_path,     index=False)
    flags.to_parquet(flags_path, index=False)
    log.info("Guardado raw:   %s  (%s filas)", raw_path,   f"{len(raw):,}")
    log.info("Guardado flags: %s  (%s filas)", flags_path, f"{len(flags):,}")

    # ── Quality report ────────────────────────────────────────────────────────
    n = len(raw)
    INDICADORES = {
        "B1_ratio":    "B1 Oferta única",
        "B2_hhi_medio":"B2 HHI por órgano",
        "B3_racha_max":"B3 Racha consecutiva",
        "B4_ratio":    "B4 Clustering umbral menor",
        "B5_ratio":    "B5 Clustering umbral SARA",
        "B6_ratio":    "B6 Procedimientos excepcionales",
    }

    qpath = REPORTS_DIR / "indicadores_comportamiento_quality.txt"
    with open(qpath, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("INFORME DE CALIDAD — INDICADORES COMPORTAMIENTO (Script 02 v2)\n")
        f.write(f"Generado: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("=" * 65 + "\n\n")

        f.write(f"Grupos únicos (adj, ccaa, cpv2d): {n:,}\n\n")

        f.write("--- Cobertura y estadísticos por indicador ---\n")
        for col, nombre in INDICADORES.items():
            if col not in raw.columns:
                f.write(f"  {nombre}: NO CALCULADO\n")
                continue
            cob  = raw[col].notna().sum()
            desc = raw[col].describe()
            f.write(f"  {nombre}:\n")
            f.write(f"    Cobertura: {cob:,} ({100*cob/n:.1f}%)\n")
            f.write(f"    Media:{desc['mean']:>10.4f}  P50:{desc['50%']:>10.4f}"
                    f"  P75:{desc['75%']:>10.4f}  Max:{desc['max']:>10.4f}\n")

        f.write("\n--- Flags binarios activados ---\n")
        for fc in ["flag_B1","flag_B2","flag_B3",
                   "flag_B4","flag_B5","flag_B6"]:
            if fc not in flags.columns:
                continue
            nf = int(flags[fc].sum())
            f.write(f"  {fc}: {nf:,} ({100*nf/n:.2f}%)\n")

        f.write("\n--- Distribución score_comportamiento (0-6) ---\n")
        for sc, cnt in flags["score_comportamiento"].value_counts().sort_index().items():
            f.write(f"  Score {sc}: {int(cnt):,} ({100*cnt/n:.1f}%)\n")

        f.write("\n--- Notas metodológicas ---\n")
        f.write("  B1 restringido a procedimiento_code en {'1','2','3'}.\n")
        f.write(f"  flag_B1: umbral fijo B1_ratio>{UMBRAL_B1_RATIO} AND B1_n>={UMBRAL_B1_N_MIN} (no p75, evita colapso bimodal).\n")
        f.write("  B4 usa es_menor_v2 (Opción D híbrida: Tier1 + código '6' < umbral).\n")
        f.write("  B5 aplica umbrales SARA por bienio y tipo de contrato.\n")
        f.write("  B6 usa procedimiento_code == '4' (negociado sin publicidad).\n")
        f.write(f"\nOutputs:\n  {raw_path}\n  {flags_path}\n")

    log.info("Quality report: %s", qpath)
    log.info("=" * 65)
    log.info("Script 02 completado. SIGUIENTE: script_03_calcular_irc.py")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
