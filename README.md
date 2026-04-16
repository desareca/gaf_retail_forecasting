# GAF Retail Forecasting

Exploración de un sistema de forecasting de ventas retail usando **Gramian Angular Fields (GAF)** como representación visual de series temporales. Convierte series de tiempo (ventas, stock, precio) en imágenes RGB y entrena un autoencoder convolucional condicionado por identidad de local y producto para predecir estados futuros, con foco en la detección de **quiebres de stock**.

> **Alcance:** Este repositorio documenta una exploración metodológica del enfoque GAF aplicado a datos retail reales. El objetivo es evaluar si la representación visual de series temporales captura patrones útiles para la detección de quiebres, no producir un sistema listo para producción.

---

## Arquitectura

```
Serie temporal (90 días)
    ventas · stock · precio
          ↓
   GAF Image (90×90×3)        R=ventas · G=stock · B=precio
          ↓
  EfficientNetB0 (encoder)    pre-entrenado ImageNet, congelado en fase 1
          ↓
   Bottleneck (3×3×1280)
          ↓
  FiLM conditioning           embedding local + producto modula el decoder
          ↓
  Decoder con upsampling      2 bloques Conv+BN+ReLU+FiLM
          ↓
  GAF predicho (14×14×3)      ventana target de 14 días
          ↓
  Diagonal invertida          forecast en valores normalizados por canal
```

Cada imagen GAF es híbrida: triángulo superior con GASF (magnitud/correlación), triángulo inferior con GADF (cambios/transiciones), y **diagonal con los valores normalizados de la serie directamente**, lo que permite reconstrucción exacta sin ambigüedad matemática.

---

## Datos

Los datos son reales, provenientes de varios retailers. Las columnas originales fueron anonimizadas para evitar fugas de información antes de versionar el repositorio.

| Campo genérico | Descripción |
|---|---|
| `date` | Fecha de la transacción |
| `store_id` | Identificador del local |
| `retailer_sku` | Identificador del producto |
| `units` | Unidades vendidas |
| `revenue` | Monto de venta |
| `stock` | Stock de cierre del día |

**Escala del dataset:**

- ~6M filas · Mar 2024 – Mar 2026
- 1,029 locales · 179 productos · 12,358 combinaciones válidas
- Split temporal estricto: Train `Mar 2024 → Ago 2025` · Val `Ago → Nov 2025` · Test `Nov 2025 → Mar 2026`

---

## Resultados

### Entrenamiento

| | Fase 1 (encoder congelado) | Fase 2 (fine-tuning) |
|---|---|---|
| Épocas | 54 | 21 |
| Mejor SSIM val | 0.6507 (época 33) | 0.6495 |
| Mejor loss val | 0.2939 | 0.2935 |

### Evaluación en Test (13,224 ventanas)

| Métrica | Valor |
|---|---|
| SSIM medio | 0.667 |
| SSIM mediana | 0.675 |
| SSIM P10 / P90 | 0.467 / 0.865 |
| MAE diagonal — Ventas (norm.) | 0.522 |
| MAE diagonal — Stock (norm.) | 0.109 |
| MAE diagonal — Precio (norm.) | 0.030 |

### Detección de quiebres de stock

Usando la diagonal predicha como señal de alerta (umbral en espacio normalizado):

| Métrica | Valor |
|---|---|
| Precisión | 0.855 |
| Recall | 0.875 |
| F1 | 0.865 |

El canal de **stock** muestra la menor error absoluto (MAE 0.109), lo que es consistente con el objetivo del sistema: el modelo captura mejor el comportamiento del stock que el de ventas, canal más ruidoso.

> **Nota sobre el período de test:** El split de test cubre Nov 2025–Mar 2026, período con efectos estacionales marcados (vacaciones de verano, Navidad) que elevan la tasa de días con ventas en cero. Esto afecta las métricas de ventas pero no invalida las de stock.

---

## Stack

- Python 3.11 · TensorFlow 2.15 · Keras (mixed precision `float16`)
- EfficientNetB0 (encoder, pesos ImageNet)
- Docker + nvidia-container-toolkit · RTX 4070
- Antigravity + Jupyter (validación visual por fase)

---

## Setup

```bash
# 1. Build
cd gaf_retail_forecasting
docker compose -f docker/docker-compose.yml build

# 2. Verificar GPU
docker compose -f docker/docker-compose.yml run --rm gaf-dev \
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 3. Levantar Jupyter
docker compose -f docker/docker-compose.yml run --rm -p 8888:8888 gaf-dev \
  jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --notebook-dir=/app/notebooks
```

Los datos van en `data/raw/` (montado como volumen, no versionado). El CSV esperado es `ventas_diarias.csv` con las columnas genéricas listadas arriba.

---

## Estructura

```
gaf_retail_forecasting/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── raw/             # Datos reales — no versionados (.gitignore)
│   └── loader.py        # Imputación, split temporal
├── gaf/
│   └── transform.py     # Construcción de imágenes GAF híbridas
├── dataset/
│   └── tf_dataset.py    # tf.data.Dataset lazy con balanceo de quiebres
├── model/
│   ├── encoder.py       # EfficientNetB0 wrapper
│   ├── decoder.py       # Upsampling + FiLM
│   ├── film.py          # Feature-wise Linear Modulation
│   ├── embedding.py     # Embeddings local + producto → MLP
│   └── autoencoder.py   # Ensamble completo
├── training/
│   └── train.py         # Loop de entrenamiento 2 fases
├── evaluation/
│   └── visualize.py     # Métricas y visualizaciones
├── notebooks/           # Validación visual por fase
├── outputs/             # Checkpoints, logs, visualizaciones — no versionados
├── mappings/            # Índices local/producto — no versionados
└── config.py            # Configuración central
```

---

## Fases

| Fase | Descripción | Estado |
|---|---|---|
| 0 | Setup Docker + GPU | ✅ |
| 1 | Transformación GAF | ✅ |
| 2 | Loader + imputación | ✅ |
| 3 | TF Dataset con balanceo | ✅ |
| 4 | Embedding + FiLM | ✅ |
| 5 | Encoder + Decoder | ✅ |
| 6 | Entrenamiento (2 fases) | ✅ |
| 7 | Evaluación en test | ✅ |
| 8 | Extensiones / mejoras | ⬜ |

---

## Decisiones de diseño relevantes

- **Normalización zero-anchored:** ventas y stock se normalizan con `min=0` fijo, de modo que stock=0 (quiebre) mapea siempre a -1 en el espacio normalizado, independiente del máximo histórico.
- **Diagonal como señal:** almacenar la serie normalizada directamente en la diagonal (no `cos(2φ)`) permite recuperar valores exactos y usar la diagonal predicha como proxy de forecast sin inversión matemática.
- **Balanceo por época:** los quiebres de stock son eventos raros (~1–5% de ventanas). El dataset de train sobremuestrea quiebres hasta un ratio configurable (`QUIEBRE_RATIO=0.40`) por época.
- **Pipeline lazy:** los GAF se generan on-the-fly vía `tf.data.Dataset.from_generator` para evitar OOM con ~6M filas.
- **FiLM gamma init:** `bias_initializer='ones'` para gamma evita activaciones muertas al inicio del entrenamiento.