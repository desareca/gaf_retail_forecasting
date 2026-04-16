from pathlib import Path

# ─── Rutas base ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data"
OUTPUTS = ROOT / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
VISUALIZATIONS = OUTPUTS / "visualizations"
LOGS = OUTPUTS / "logs"
MAPPINGS = ROOT / "mappings"
NOTEBOOKS = ROOT / "notebooks"

# ─── Parámetros GAF ───────────────────────────────────────────────────────────
WINDOW_SIZE      = 90   # días de historia → imagen 90x90
FORECAST_HORIZON = 14    # días a predecir → ventana target
SEED = 42

# ─── Canales de la imagen GAF ─────────────────────────────────────────────────
GAF_CHANNELS = ["unidades", "stock", "precio"]  # orden RGB
IMAGE_SIZE = (WINDOW_SIZE, WINDOW_SIZE, len(GAF_CHANNELS))  # (90, 90, 3)

# ─── Split temporal ───────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.85   # 70%–85% → TEST = últimas 15%

# ─── Entrenamiento ────────────────────────────────────────────────────────────
BATCH_SIZE              = 16
EPOCHS_PHASE1           = 100
EPOCHS_PHASE2           = 50
LR_PHASE1               = 1e-3
LR_PHASE2               = 5e-5
EARLY_STOPPING_PATIENCE = 20
N_COMBOS_PER_EPOCH      = 4000 # 6400  # combinaciones sorteadas por época (train)
N_COMBOS_PER_EPOCH_VAL  = 1000 # 1600   # combinaciones sorteadas por época (val)

# ─── Filtrado y balanceo de ventanas ──────────────────────────────────────────
# Quiebre definido como: al menos 1 día con stock_cierre == 0 en el target.
# Una semana sin ventas puede ser normal (baja rotación, festivo, etc.),
# pero stock == 0 indica producto no disponible — quiebre operacional real.

# Filtro de input: descarta ventanas donde el stock estuvo en cero
# más de este porcentaje del tiempo (combinación probablemente descontinuada).
INPUT_STOCK_CERO_UMBRAL = 0.60  # 60% → si >60% de los 90 días tiene stock=0, se descarta

# Balanceo por época (solo train): proporción de ventanas con quiebre de stock
# en el target. El resto se completa con ventanas sin quiebre.
# 0.40 → 40% quiebres, 60% normales por época.
QUIEBRE_RATIO = 0.40

# ─── Imputación ───────────────────────────────────────────────────────────────
IMPUTE_WINDOW_DAYS = 7    # promedio rolling para unidades/monto
PRICE_FFILL_DAYS   = 14   # forward fill máximo para precio

# ─── Embeddings ───────────────────────────────────────────────────────────────
NUM_LOCALES      = 1200
NUM_PRODUCTOS    = 200

LOCAL_EMB_DIM    = 32
PRODUCTO_EMB_DIM = 64
EMB_MLP_HIDDEN   = 128
EMB_OUTPUT_DIM   = 64
EMB_DROPOUT      = 0.1

# ─── Loss ─────────────────────────────────────────────────────────────────────
LOSS_MSE_WEIGHT  = 0.6
LOSS_SSIM_WEIGHT = 0.15
LOSS_TEMP_WEIGHT = 0.25

LOSS_VAR_WEIGHT = 0.3

# ─── Pesos por canal en la loss ───────────────────────────────────────────────
# Orden: [ventas, stock, precio]
LOSS_CHANNEL_WEIGHTS = [0.5, 0.45, 0.05]