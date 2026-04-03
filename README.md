# Análisis estadístico de señales de riesgo en contratación pública española (2019–2025)

**Trabajo de Fin de Grado — Grado en Comercio**  
**Autor:** Jan Calsina Varela  
**Universidad:** [Universidad Complutense de Madrid]  
**Tutor:** [Arturo Benayas Ayuso]  
**Año:** 2026

---

## Descripción

Este repositorio contiene los scripts de análisis y visualización desarrollados para el TFG *"Análisis estadístico de señales de riesgo en contratación pública española (2019–2025)"*.

El trabajo construye un **Índice de Riesgo de Contratación (IRC)** sobre un corpus de más de 9,2 millones de registros de adjudicación pública, integrando datos de la Plataforma de Contratación del Sector Público (PLACSP), seis portales autonómicos, el diario oficial europeo TED y el Registro Mercantil (BORME).

El enfoque es estrictamente estadístico y no acusatorio: los indicadores identifican señales y anomalías estadísticas, sin implicar responsabilidad legal de ninguna empresa o entidad.

---

## Estructura del repositorio
tfg-riesgo-contratacion-publica/
│
│
├── pipeline/
│   ├── script_01_calcular_flags_corporativos.py
│   ├── script_02_calcular_indicadores_comportamiento.py
│   ├── script_03_calcular_irc.py
│   ├── script_04_tablas_descriptivas.py
│   ├── script_05_mapa_territorio_sector.py
│   ├── script_06_analisis_bivariado.py
│   ├── script_07_modelos_regresion.py
│   └── script_08_validaciones_robustez.py
│
├── visualization/
│   ├── figura1_distribucion_irc.py
│   ├── figura2_irc_territorial.py
│   ├── figura3_irc_procedimiento.py
│   ├── figura4_prevalencia_flags.py
│   ├── figura5_scatter_irc_importe.py
│   └── figura6_estabilidad_temporal.py
│
│
├── requirements.txt                           # Dependencias Python
└── README.md

---

## Datos

Los datos **no se incluyen en este repositorio** por su volumen (>9,2 M registros en formato Apache Parquet).

Las fuentes originales son:

| Fuente | Descripción | Repositorio original |
|--------|-------------|----------------------|
| PLACSP | Plataforma de Contratación del Sector Público | [licitaciones-espana](https://github.com/BquantFinance/licitaciones-espana) |
| BORME | Registro Mercantil (actos y cargos societarios) | [licitaciones-espana](https://github.com/BquantFinance/licitaciones-espana) |
| TED | Tenders Electronic Daily (contratos SARA) | [licitaciones-espana](https://github.com/BquantFinance/licitaciones-espana) |
| CCAA | Andalucía, Cataluña, Euskadi, Galicia, Madrid, Valencia | [licitaciones-espana](https://github.com/BquantFinance/licitaciones-espana) |

Los datos procesados se encuentran en formato Parquet en soporte externo local con la siguiente estructura:
Base De Datos TFG JAN/
├── Base de Datos/
│   ├── CCAA/          ← Datos autonómicos curados
│   ├── Nacional/      ← PLACSP, TED, BORME
│   └── reports/       ← Outputs del pipeline (inputs para visualización)
└── Documentacion/

Para reproducir el análisis, descarga los datos desde el repositorio de origen y ajusta las rutas en cada script a tu ruta local.

---

## Instalación
```bash
git clone https://github.com/jcalsinav/tfg-riesgo-contratacion-publica.git
cd tfg-riesgo-contratacion-publica
pip install -r requirements.txt
```

---

## Uso

Ejecuta los scripts del pipeline en orden estricto (01 → 08), ya que cada script consume los outputs del anterior:
```bash
python scripts/pipeline/01_borme_placsp_match.py
python scripts/pipeline/02_indicadores_comportamiento.py
python scripts/pipeline/03_irc_construccion.py
# ... y así sucesivamente hasta el 08
```

Una vez completado el pipeline, genera las figuras:
```bash
python scripts/visualization/figura1_distribucion_irc.py
python scripts/visualization/figura2_irc_territorial.py
# ... etc.
```

---

## Dependencias
pandas>=1.5
pyarrow>=10.0
statsmodels>=0.13
scipy>=1.9
matplotlib>=3.6
seaborn>=0.12
numpy>=1.23
geopandas>=0.12

---

## Indicadores del IRC

### Bloque corporativo (BORME)
| Flag | Descripción |
|------|-------------|
| F1 | Empresa de constitución reciente (<6 meses antes de primera adjudicación) |
| F2 | Capital social <10.000€ con importe acumulado >100.000€ |
| F3 | Administradores compartidos entre empresas adjudicatarias del mismo órgano |
| F4 | Disolución en <12 meses desde la última adjudicación |
| F5 | Empresa en situación concursal activa |

### Bloque de comportamiento (PLACSP + CCAA)
| Flag | Descripción |
|------|-------------|
| B1 | Proporción de adjudicaciones con oferta única >50% (mín. 3 contratos) |
| B2 | Índice HHI por órgano contratante en el percentil 75 del grupo de referencia |
| B3 | Racha de adjudicaciones consecutivas al mismo NIF sin competencia |
| B4 | Clustering de importes en banda [13.500–15.000€] o [34.000–40.000€] (>20%) |
| B5 | Clustering de importes en el 10% inferior al umbral SARA aplicable (>20%) |
| B6 | Tasa de contratos no menores adjudicados sin publicidad >P75 del grupo |

---

## Resultados principales

- **Corpus**: 9.222.246 registros · 477.926 adjudicatarios únicos · 1.033.110 grupos NIF×CCAA×CPV
- **IRC medio**: 0,0612 · P75: 0,10 · Máximo: 0,66
- **Factor con mayor efecto sobre IRC**: tipo de procedimiento (η²=0,1934, efecto GRANDE)
- **Estabilidad temporal**: ρ=0,8070 entre subperíodos 2019–2021 y 2022–2025 (umbral ≥0,70)
- **Predictor principal**: β(B1_ratio)=0,01087 > β(log_importe)=0,00961 en modelo MCO

---

## Referencia de datos originales

Sánchez Vidal, G. (2024). *licitaciones-espana: Dataset unificado de contratación pública española*. BQuant Finance. https://github.com/BquantFinance/licitaciones-espana

---

## Licencia

MIT License. Ver [LICENSE](LICENSE) para más detalles.

> **Nota ética**: Este trabajo opera exclusivamente con datos públicos bajo licencias de reutilización. Todos los hallazgos son estadísticos y no implican responsabilidad legal de ninguna entidad. Los análisis BORME operan sobre datos agregados sin referencia a personas físicas identificables.
