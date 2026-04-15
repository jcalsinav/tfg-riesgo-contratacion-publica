# Procurement Risk Index for Spanish Public Contracting (2019–2025)
# Índice de Riesgo de Contratación en la Contratación Pública Española (2019–2025)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Data: Open](https://img.shields.io/badge/data-open%20access-green.svg)](https://github.com/BquantFinance/licitaciones-espana)

---

## English

### Overview

This repository contains the full analytical pipeline for the Bachelor's thesis:

> **"Análisis estadístico de señales de riesgo en contratación pública española (2019–2025)"**  
> Jan Calsina Varela · Universidad Complutense de Madrid · 2026  
> Supervisor: Arturo Benayas Ayuso

The project constructs and validates a **Procurement Risk Index (IRC)** over a corpus of **9,222,246 procurement records** from seven Spanish public contracting sources, covering the period 2019–2025. The index integrates eleven indicators across two blocks: five corporate indicators (derived from the Commercial Registry / BORME) and six behavioral indicators (derived from award records). The analytical unit is taxpayer ID × autonomous community × CPV sector, yielding 1,033,110 groups.

### Key findings

- **Contract type** is the dominant predictor of procurement risk (η²=0.2116, large effect). The autonomous community explains only 0.6% of index variance (η²=0.006, trivial).
- **HHI concentration (B2)** activates in 20.07% of groups with a homogeneous distribution across all regions (23–29%), indicating a systemic market characteristic rather than a territorial anomaly.
- **Temporal stability** of the index is robust (Spearman ρ=0.7880 between sub-periods).
- **Network analysis (F3)** identifies 1,587 award-receiving companies connected through 1,863 shared administrators, with the main component (660 companies) concentrating €10.7 billion in high-IRC awards.
- **Strategic sectors** (CPV 09, 31, 32, 35) show an IRC mean 18% above the general corpus, but with trivial effect size (η²≤0.0005). The median IRC in strategic sectors (0.080) is substantially higher than the general corpus median (0.000), indicating near-universal signal activation.

### Pipeline diagram
[Raw data sources]
PLACSP · TED · BORME · AN · CT · PV · GA · MD · VC
│
▼
[Scripts 01–02: Indicator construction]
script_01 → Corporate flags F1–F5 (BORME × adjudicatarios)
script_02 → Behavioral flags B1–B6 (NIF×CCAA×CPV groups)
│
▼
[Script 03: IRC construction]
IRC = 0.4·Sc + 0.6·SB
4 variants: irc_base, irc_iguales, irc_comp, irc_corp
│
▼
[Scripts 04–05: Aggregation]
Territorial · Sectoral · Temporal
│
▼
[Script 06: Bivariate analysis]
Spearman correlations · Kruskal-Wallis · effect sizes
│
▼
[Script 07: Econometric modelling]
OLS + Logit · Fixed effects (sector × CCAA × year)
│
▼
[Script 08: Robustness validation]
V01 weight sensitivity · V02 threshold sensitivity
V03 temporal stability · V04 TED coherence
│
▼
[Script 09: Network analysis]
F3 administrator graph · centrality · cluster detection
│
▼
[Script 10: Strategic sectors]
CPV 09·31·32·35 vs. general corpus
│
▼
[Visualization scripts × 6]
Figures 1–6 + Annex B·C figures

### Repository structure
tfg-riesgo-contratacion-publica/
│
├── pipeline/
│   ├── script_01_adjudicatarios_flags.py
│   ├── script_02_comportamiento_flags.py
│   ├── script_03_irc_construccion.py
│   ├── script_04_agregacion.py
│   ├── script_05_descriptivo_territorial.py
│   ├── script_06_bivariado.py
│   ├── script_07_econometrico.py
│   ├── script_08_robustez.py
│   ├── script_09_network_analysis.py
│   └── script_10_sectores_estrategicos.py
│
├── visualization/
│   ├── viz_01_flags_prevalencia.py
│   ├── viz_02_irc_distribucion.py
│   ├── viz_03_irc_territorial.py
│   ├── viz_04_procedimiento_boxplot.py
│   ├── viz_05_importes_scatter.py
│   └── viz_06_concentracion_robustez.py
│
├── config.py                  ← set your local data path here
├── requirements.txt
└── README.md

### Reproduction instructions

**1. Clone the repository**
```bash
git clone https://github.com/jcalsinav/tfg-riesgo-contratacion-publica
cd tfg-riesgo-contratacion-publica
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the data**

Input data is not distributed in this repository due to size. Download the curated Parquet files from the BQuant Finance public repository:

→ [https://github.com/BquantFinance/licitaciones-espana/releases/latest](https://github.com/BquantFinance/licitaciones-espana/releases/latest)

Required files:
- `nacional.zip` (PLACSP)
- `andalucia.zip`
- `catalunya.zip`
- `euskadi.zip`
- `galicia.zip`
- `comunidad_madrid.zip`
- `valencia.zip`
- `borme.zip`
- `ted.zip`

**4. Configure the data path**

Edit `config.py` and set `BASE_DIR` to your local data directory:
```python
BASE_DIR = "/your/path/to/Base De Datos"
```

**5. Run the pipeline**
```bash
python pipeline/script_01_adjudicatarios_flags.py
python pipeline/script_02_comportamiento_flags.py
python pipeline/script_03_irc_construccion.py
python pipeline/script_04_agregacion.py
python pipeline/script_05_descriptivo_territorial.py
python pipeline/script_06_bivariado.py
python pipeline/script_07_econometrico.py
python pipeline/script_08_robustez.py
python pipeline/script_09_network_analysis.py
python pipeline/script_10_sectores_estrategicos.py
```

Each script generates a quality report (`*_quality_TIMESTAMP.txt`) in the `reports/` output directory. Scripts must be run in order. Estimated total runtime: 45–90 minutes on a standard laptop with data on external storage.

**6. Generate figures**
```bash
python visualization/viz_01_flags_prevalencia.py
# ... repeat for viz_02 through viz_06
```

### Use cases

**Risk monitoring**  
The IRC pipeline can be re-run periodically as new procurement data is published (BQuant updates quarterly). Groups crossing the two-flag threshold (5.9% of the corpus) generate a prioritized watchlist for supervisory bodies without requiring manual case selection.

**Integrity analysis**  
The network analysis module (script_09) identifies clusters of award-receiving companies sharing administrators. These clusters can be used as input for qualitative review of specific procurement segments, particularly in strategic sectors (CPV 09, 31, 32, 35).

**Benchmarking across regions**  
The territorial aggregation (scripts 04–05) produces comparable IRC rankings across autonomous communities. Researchers can extend coverage to additional regional portals by adding a source script following the existing pipeline conventions.

**Policy evaluation**  
The temporal stability module (script_08, V03) allows comparison of IRC distributions across sub-periods. This can be used to evaluate whether regulatory changes (e.g. LCSP 2017 amendments) produced measurable effects on risk signal patterns.

### Technical requirements

| Component | Specification |
|---|---|
| Language | Python 3.10+ |
| Key libraries | pandas, numpy, statsmodels, scipy, networkx, matplotlib, seaborn |
| Data format | Apache Parquet (columnar binary) |
| OS | macOS / Linux (Windows compatible with path adjustments) |

### Citation
Calsina Varela, J. (2026). Análisis estadístico de señales de riesgo en
contratación pública española (2019–2025) [Bachelor's thesis].
Universidad Complutense de Madrid.
https://github.com/jcalsinav/tfg-riesgo-contratacion-publica

### License

MIT License. Input data is subject to the open data reuse conditions of the Spanish Government ([datos.gob.es](https://datos.gob.es/es/aviso-legal)) and the EU Open Data Licence ([data.europa.eu](https://data.europa.eu/eli/dec_impl/2011/833/oj)).

---

## Español

### Descripción

Este repositorio contiene el pipeline analítico completo del Trabajo de Fin de Grado:

> **"Análisis estadístico de señales de riesgo en contratación pública española (2019–2025)"**  
> Jan Calsina Varela · Universidad Complutense de Madrid · 2026  
> Tutor: Arturo Benayas Ayuso

El proyecto construye y valida un **Índice de Riesgo de Contratación (IRC)** sobre un corpus de **9.222.246 registros** procedentes de siete fuentes de contratación pública española, para el período 2019–2025. El índice integra once indicadores en dos bloques: cinco corporativos (a partir del BORME) y seis de comportamiento licitatorio (a partir de los registros de adjudicación). La unidad de análisis es NIF adjudicatario × comunidad autónoma × sector CPV, con 1.033.110 grupos.

### Hallazgos principales

- El **tipo de contrato** es el predictor dominante del riesgo (η²=0,2116, efecto grande). La comunidad autónoma explica solo el 0,6% de la varianza del IRC (efecto trivial).
- El indicador de **concentración HHI (B2)** se activa en el 20,07% de los grupos con distribución homogénea entre territorios (23–29%), lo que indica una característica sistémica del mercado.
- La **estabilidad temporal** del índice es robusta (ρ de Spearman=0,7880 entre subperíodos).
- El **análisis de redes (F3)** identifica 1.587 empresas adjudicatarias conectadas por 1.863 administradores compartidos; el componente principal (660 empresas) concentra 10.699 M€ en adjudicaciones con IRC alto.
- Los **sectores estratégicos** (CPV 09, 31, 32, 35) presentan un IRC medio un 18% superior al corpus general, con tamaño del efecto trivial pero mediana desplazada (0,080 vs 0,000).

### Instrucciones de reproducción

Ver sección en inglés (idéntico procedimiento). Ajustar la ruta local en `config.py`.

### Casos de uso

**Monitorización de riesgos** — El pipeline puede re-ejecutarse periódicamente conforme se publican nuevos datos. Los grupos que cruzan el umbral de dos flags simultáneos generan una lista de priorización para organismos supervisores.

**Análisis de integridad** — El módulo de análisis de redes (script_09) identifica clusters de empresas adjudicatarias con administradores comunes, utilizables como input para revisión cualitativa en sectores estratégicos.

**Evaluación de políticas** — El módulo de estabilidad temporal (script_08) permite comparar distribuciones del IRC entre subperíodos para evaluar el efecto de cambios normativos.

### Cita
Calsina Varela, J. (2026). Análisis estadístico de señales de riesgo en
contratación pública española (2019–2025) [Trabajo de Fin de Grado].
Universidad Complutense de Madrid.
https://github.com/jcalsinav/tfg-riesgo-contratacion-publica
