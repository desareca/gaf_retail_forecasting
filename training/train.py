# training/train.py
import os
import sys
sys.path.append('/app')

import numpy as np
import tensorflow as tf
import json
from tensorflow import keras
from tensorflow.keras import mixed_precision

from config import (
    BATCH_SIZE, EPOCHS_PHASE1, EPOCHS_PHASE2,
    LR_PHASE1, LR_PHASE2, EARLY_STOPPING_PATIENCE,
    LOSS_MSE_WEIGHT, LOSS_SSIM_WEIGHT, LOSS_CHANNEL_WEIGHTS, 
    CHECKPOINTS, LOGS, FORECAST_HORIZON, LOSS_VAR_WEIGHT, LOSS_TEMP_WEIGHT,
    EMB_OUTPUT_DIM, N_COMBOS_PER_EPOCH, N_COMBOS_PER_EPOCH_VAL,
    NUM_LOCALES, NUM_PRODUCTOS, SEED, QUIEBRE_RATIO,
)
from model.autoencoder import build_autoencoder
from model.encoder import unfreeze_top_blocks


# ── Mixed precision (RTX 4070 soporta float16) ────────────────────────────────
mixed_precision.set_global_policy("mixed_float16")


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
# Pesos por canal como constante TF para uso dentro del grafo
_CHANNEL_WEIGHTS = tf.constant(LOSS_CHANNEL_WEIGHTS, dtype=tf.float32)  # [0.5, 0.4, 0.1]

def ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    SSIM con filtro 3×3 para imágenes FORECAST_HORIZON×FORECAST_HORIZON, pesado por canal.
    """
    y_true_f32 = tf.cast(y_true, tf.float32)
    y_pred_f32 = tf.cast(y_pred, tf.float32)
    y_true_01  = (y_true_f32 + 1.0) / 2.0
    y_pred_01  = (y_pred_f32 + 1.0) / 2.0

    # SSIM por canal: (batch, C)
    ssim_per_channel = tf.stack([
        tf.image.ssim(
            y_true_01[..., c:c+1],
            y_pred_01[..., c:c+1],
            max_val=1.0,
            filter_size=3,
            filter_sigma=0.5,
            k1=0.01,
            k2=0.03,
        )
        for c in range(3)
    ], axis=-1)  # (batch, 3)

    # Promedio por batch y peso por canal
    ssim_mean = tf.reduce_mean(ssim_per_channel, axis=0)  # (3,)
    ssim_weighted = tf.reduce_sum(ssim_mean * _CHANNEL_WEIGHTS)
    return 1.0 - ssim_weighted

def mae_diagonal_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MAE solo sobre la diagonal, pesado por canal."""
    # Extraer diagonal: (batch, 3, 7)
    diag_true = tf.linalg.diag_part(tf.transpose(y_true, [0, 3, 1, 2]))
    diag_pred = tf.linalg.diag_part(tf.transpose(y_pred, [0, 3, 1, 2]))

    # MSE por canal: (3,)
    mae_per_channel = tf.reduce_mean(tf.abs(diag_true - diag_pred), axis=[0, 2])

    # Peso por canal
    return tf.reduce_sum(mae_per_channel * _CHANNEL_WEIGHTS)

def mse_diagonal_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MSE solo sobre la diagonal, pesado por canal."""
    # Extraer diagonal: (batch, 3, 7)
    diag_true = tf.linalg.diag_part(tf.transpose(y_true, [0, 3, 1, 2]))
    diag_pred = tf.linalg.diag_part(tf.transpose(y_pred, [0, 3, 1, 2]))

    # MSE por canal: (3,)
    mse_per_channel = tf.reduce_mean(tf.square(diag_true - diag_pred), axis=[0, 2])

    # Peso por canal
    return tf.reduce_sum(mse_per_channel * _CHANNEL_WEIGHTS)

def robust_diagonal_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    MSE + Penalización de Varianza sobre la diagonal, pesado por canal.
    Evita que la predicción colapse al promedio.
    """
    # 1. Extraer diagonal por canal: (batch, channels, diag_len)
    # Transposición para llevar los canales a la dim 1 antes de sacar la diagonal
    diag_true = tf.linalg.diag_part(tf.transpose(y_true, [0, 3, 1, 2]))
    diag_pred = tf.linalg.diag_part(tf.transpose(y_pred, [0, 3, 1, 2]))

    # 2. MSE por canal: (channels,)
    # Promediamos sobre el batch (0) y el largo de la diagonal (2)
    mae_per_channel = tf.reduce_mean(tf.abs(diag_true - diag_pred), axis=[0, 2])

    # 3. Penalización de Varianza (Anti-Promedio): (channels,)
    # Calculamos qué tan "dinámica" es la serie en la diagonal
    var_true = tf.math.reduce_variance(diag_true, axis=2) # (batch, channels)
    var_pred = tf.math.reduce_variance(diag_pred, axis=2) # (batch, channels)
    
    # Comparamos la varianza real vs la predicha
    # Promediamos sobre el batch para obtener un error de varianza por canal
    variance_loss_per_channel = tf.reduce_mean(tf.square(var_true - var_pred), axis=0)

    # 4. Combinación y Pesos
    # Sumamos ambos errores antes de aplicar los pesos específicos por canal
    # Nota: Puedes multiplicar variance_loss por un escalar (lambda) si quieres 
    # que la penalización sea más o menos agresiva.
    total_per_channel = mae_per_channel * (1 - LOSS_VAR_WEIGHT) + variance_loss_per_channel * LOSS_VAR_WEIGHT
    
    return tf.reduce_sum(total_per_channel * _CHANNEL_WEIGHTS)

def temporal_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Penaliza cuando la dirección temporal predicha difiere de la real.
    Opera sobre la diagonal de cada imagen GAF.
    """
    # Extraer diagonal: (batch, 3, 14)
    diag_true = tf.linalg.diag_part(tf.transpose(y_true, [0, 3, 1, 2]))
    diag_pred = tf.linalg.diag_part(tf.transpose(y_pred, [0, 3, 1, 2]))

    # Diferencias entre días consecutivos: (batch, 3, 13)
    diff_true = diag_true[:, :, 1:] - diag_true[:, :, :-1]
    diff_pred = diag_pred[:, :, 1:] - diag_pred[:, :, :-1]

    # MAE sobre las diferencias — penaliza error en la dirección/magnitud del cambio
    mae_diff = tf.reduce_mean(tf.abs(diff_true - diff_pred), axis=[0, 2])  # (3,)

    return tf.reduce_sum(mae_diff * _CHANNEL_WEIGHTS)


def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_f32 = tf.cast(y_true, tf.float32)
    y_pred_f32 = tf.cast(y_pred, tf.float32)

    #mae_diag = mae_diagonal_loss(y_true_f32, y_pred_f32)
    #mse_diag = mse_diagonal_loss(y_true_f32, y_pred_f32)
    mae_var_diag = robust_diagonal_loss(y_true_f32, y_pred_f32)
    ssim = ssim_loss(y_true_f32, y_pred_f32)
    temporal = temporal_loss(y_true_f32, y_pred_f32)

    return LOSS_MSE_WEIGHT * mae_var_diag + LOSS_SSIM_WEIGHT * ssim + LOSS_TEMP_WEIGHT * temporal


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def ssim_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """SSIM promedio sobre todos los canales (sin pesos — para monitoreo)."""
    y_true_f32 = tf.cast(y_true, tf.float32)
    y_pred_f32 = tf.cast(y_pred, tf.float32)
    y_true_01  = (y_true_f32 + 1.0) / 2.0
    y_pred_01  = (y_pred_f32 + 1.0) / 2.0
    return tf.reduce_mean(tf.image.ssim(
        y_true_01, y_pred_01,
        max_val=1.0,
        filter_size=3,
        filter_sigma=0.5,
        k1=0.01,
        k2=0.03,
    ))


def mae_diagonal_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    MAE solo sobre la diagonal principal de la imagen GAF FORECAST_HORIZON×FORECAST_HORIZON.
    La diagonal = serie de tiempo original desnormalizada.
    Solo mide lo que importa para el forecast.
    """
    y_true_f32 = tf.cast(y_true, tf.float32)
    y_pred_f32 = tf.cast(y_pred, tf.float32)

    # Extraer diagonal: (batch, 7, 7, 3) → (batch, 7, 3)
    diag_true = tf.linalg.diag_part(
        tf.transpose(y_true_f32, [0, 3, 1, 2])  # (batch, 3, 7, 7)
    )  # (batch, 3, 7)
    diag_pred = tf.linalg.diag_part(
        tf.transpose(y_pred_f32, [0, 3, 1, 2])
    )  # (batch, 3, 7)

    return tf.reduce_mean(tf.abs(diag_true - diag_pred))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset sintético para smoke test
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_dataset(n_samples: int, batch_size: int) -> tf.data.Dataset:
    """
    Genera un dataset sintético para verificar que el pipeline no tiene errores.
    Reemplazar con el dataset real en Fase 8.
    """
    np.random.seed(42)
    gaf_inputs   = np.random.uniform(-1, 1, (n_samples, 90, 90, 3)).astype("float32")
    local_idxs   = np.random.randint(0, NUM_LOCALES,   size=(n_samples,)).astype("int32")
    prod_idxs    = np.random.randint(0, NUM_PRODUCTOS,  size=(n_samples,)).astype("int32")
    gaf_targets  = np.random.uniform(-1, 1, (n_samples, FORECAST_HORIZON,  FORECAST_HORIZON,  3)).astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((
        {"gaf_input": gaf_inputs, "local_idx": local_idxs, "producto_idx": prod_idxs},
        gaf_targets,
    ))
    return ds.shuffle(512).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def make_real_datasets(n_combos: int = N_COMBOS_PER_EPOCH):
    """
    Carga datos reales y construye datasets de train y val
    con sampleo de combinaciones por época.
    """
    import pandas as pd
    from data.loader import load_and_prepare, COL_FECHA
    from dataset.tf_dataset import build_tf_dataset_sampled

    print("Cargando datos reales...")
    train, val, test, local_map, product_map, valid_combos = load_and_prepare()

    df_full   = pd.concat([train, val, test])
    train_end = train[COL_FECHA].max()
    val_end   = val[COL_FECHA].max()

    print(f"Construyendo dataset train ({n_combos} combinaciones)...")
    train_ds = build_tf_dataset_sampled(
        df_full, local_map, product_map, valid_combos,
        split="train",
        train_end_date=train_end,
        val_end_date=val_end,
        batch_size=BATCH_SIZE,
        shuffle=True,
        n_combos=n_combos,
    )

    print(f"Construyendo dataset val ({N_COMBOS_PER_EPOCH_VAL} combinaciones)...")
    val_ds = build_tf_dataset_sampled(
        df_full, local_map, product_map, valid_combos,
        split="val",
        train_end_date=train_end,
        val_end_date=val_end,
        batch_size=BATCH_SIZE,
        shuffle=False,
        n_combos=N_COMBOS_PER_EPOCH_VAL, # 1000,
    )

    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def make_callbacks(phase: int) -> list:
    os.makedirs(CHECKPOINTS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINTS / f"phase{phase}_best.weights.h5"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=str(LOGS / f"phase{phase}"),
        histogram_freq=0,
        update_freq="epoch",
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )
    return [checkpoint, early_stop, tensorboard, reduce_lr]


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def compile_model(autoencoder: keras.Model, lr: float) -> None:
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=combined_loss,
        metrics=[ssim_metric, mae_diagonal_metric],
    )


def train(
    df_full,
    local_map: dict,
    product_map: dict,
    valid_combos,
    train_end,
    val_end,
    smoke_test: bool = False,
) -> dict:
    from dataset.tf_dataset import build_tf_dataset_sampled

    epochs1 = 2 if smoke_test else EPOCHS_PHASE1
    epochs2 = 2 if smoke_test else EPOCHS_PHASE2

    if smoke_test:
        print("\n⚠️  SMOKE TEST — datos sintéticos | sin checkpoints | sin logs")
        _smoke_train_ds = make_synthetic_dataset(256, BATCH_SIZE)
        _smoke_val_ds   = make_synthetic_dataset(64,  BATCH_SIZE)


    autoencoder, encoder, decoder, emb_model = build_autoencoder(encoder_trainable=False)

    # ── Fase 1: encoder congelado ─────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  FASE 1 — Encoder congelado")
    print(f"  LR={LR_PHASE1}  |  Épocas={epochs1}")
    print("═" * 60)

    compile_model(autoencoder, lr=LR_PHASE1)

    best_loss = float('inf')
    best_ssim = 0.0
    patience_counter = 0
    history1 = {"loss": [], "val_loss": [], "ssim_metric": [], "val_ssim_metric": [],
    "mae_diagonal_metric": [], "val_mae_diagonal_metric": []}

    # Writers de TensorBoard
    os.makedirs(str(LOGS / "phase1"), exist_ok=True)
    os.makedirs(str(LOGS / "phase2"), exist_ok=True)
    writer1 = tf.summary.create_file_writer(str(LOGS / "phase1"))
    writer2 = tf.summary.create_file_writer(str(LOGS / "phase2"))

    for epoch in range(epochs1):
        # Re-sampleo: cada época ve combinaciones distintas
        if smoke_test:
            train_ds = _smoke_train_ds
            val_ds   = _smoke_val_ds
        else:
            seed_epoch = SEED + epoch
            train_ds = build_tf_dataset_sampled(
                df_full, local_map, product_map, valid_combos,
                split="train",
                train_end_date=train_end,
                val_end_date=val_end,
                batch_size=BATCH_SIZE,
                shuffle=True,
                n_combos=N_COMBOS_PER_EPOCH,
                seed=seed_epoch,
                quiebre_ratio=QUIEBRE_RATIO,
            )
            val_ds = build_tf_dataset_sampled(
                df_full, local_map, product_map, valid_combos,
                split="val",
                train_end_date=train_end,
                val_end_date=val_end,
                batch_size=BATCH_SIZE,
                shuffle=False,
                n_combos=N_COMBOS_PER_EPOCH_VAL,
                seed=SEED,  # val siempre igual para comparabilidad
            )

        result = autoencoder.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1,
            verbose=1,
        )

        # Métricas
        loss     = result.history["loss"][0]
        val_loss = result.history["val_loss"][0]
        ssim     = result.history["ssim_metric"][0]
        val_ssim = result.history["val_ssim_metric"][0]
        mae      = result.history["mae_diagonal_metric"][0]
        val_mae  = result.history["val_mae_diagonal_metric"][0]

        for k, v in zip(history1.keys(), [loss, val_loss, ssim, val_ssim, mae, val_mae]):
            history1[k].append(v)

        print(f"Época {epoch+1}/{epochs1} — loss:{loss:.4f} val_loss:{val_loss:.4f} "
              f"SSIM:{ssim:.4f} val_SSIM:{val_ssim:.4f}")

        # Checkpoint manual
        if val_loss < best_loss:
            best_loss = val_loss
            best_ssim = val_ssim
            if not smoke_test:
                autoencoder.save_weights(str(CHECKPOINTS / "phase1_best.weights.h5"))
                print(f"  ✅ Nuevo mejor Loss val: {best_loss:.4f} — checkpoint guardado")
            else:
                print(f"  ✅ Nuevo mejor Loss val: {best_loss:.4f} (smoke — no se guarda)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Sin mejora ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping en época {epoch+1}")
                break

        # ReduceLR manual
        if patience_counter > 0 and patience_counter % 5 == 0:
            old_lr = float(autoencoder.optimizer.learning_rate)
            new_lr = max(old_lr * 0.5, 1e-5)
            autoencoder.optimizer.learning_rate.assign(new_lr)
            print(f"  LR reducido: {old_lr:.2e} → {new_lr:.2e}")

        # Log a TensorBoard
        if not smoke_test:
            with writer1.as_default():
                tf.summary.scalar("loss",            loss,     step=epoch)
                tf.summary.scalar("val_loss",        val_loss, step=epoch)
                tf.summary.scalar("ssim_metric",     ssim,     step=epoch)
                tf.summary.scalar("val_ssim_metric", val_ssim, step=epoch)
                tf.summary.scalar("mae_metric",      mae,      step=epoch)
                tf.summary.scalar("val_mae_metric",  val_mae,  step=epoch)
                tf.summary.scalar("lr", float(autoencoder.optimizer.learning_rate), step=epoch)
            writer1.flush()

    # Cargar mejores pesos
    if not smoke_test:
        autoencoder.load_weights(str(CHECKPOINTS / "phase1_best.weights.h5"),
                             skip_mismatch=True)

    final_loss = best_loss
    print(f"\n  Loss validación Fase 1 (mejor): {final_loss:.4f} | SSIM validación Fase 1: {best_ssim:.4f}")
    if not smoke_test:
        if best_ssim < 0.7:
            print("  ⚠️  SSIM < 0.7 — considera más épocas antes de fine-tuning")
        else:
            print("  ✅ Objetivo SSIM > 0.7 alcanzado")

    # ── Fase 2: fine-tuning ───────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  FASE 2 — Fine-tuning encoder (últimos 3 bloques)")
    print(f"  LR={LR_PHASE2}  |  Épocas={epochs2}")
    print("═" * 60)

    unfreeze_top_blocks(encoder, num_blocks=3)
    compile_model(autoencoder, lr=LR_PHASE2)

    best_ssim2 = 0.0
    best_loss2 = float('inf')
    patience_counter2 = 0
    history2 = {"loss": [], "val_loss": [], "ssim_metric": [], "val_ssim_metric": [],
                "mae_diagonal_metric": [], "val_mae_diagonal_metric": []}

    for epoch in range(epochs2):
        if smoke_test:
            train_ds = _smoke_train_ds
            val_ds   = _smoke_val_ds
        else:
            seed_epoch = SEED + 1000 + epoch
            train_ds = build_tf_dataset_sampled(
                df_full, local_map, product_map, valid_combos,
                split="train",
                train_end_date=train_end,
                val_end_date=val_end,
                batch_size=BATCH_SIZE,
                shuffle=True,
                n_combos=N_COMBOS_PER_EPOCH,
                seed=seed_epoch,
                quiebre_ratio=QUIEBRE_RATIO,
            )
            val_ds = build_tf_dataset_sampled(
                df_full, local_map, product_map, valid_combos,
                split="val",
                train_end_date=train_end,
                val_end_date=val_end,
                batch_size=BATCH_SIZE,
                shuffle=False,
                n_combos=N_COMBOS_PER_EPOCH_VAL,
                seed=SEED,
            )

        result = autoencoder.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)

        loss     = result.history["loss"][0]
        val_loss = result.history["val_loss"][0]
        ssim     = result.history["ssim_metric"][0]
        val_ssim = result.history["val_ssim_metric"][0]
        mae      = result.history["mae_diagonal_metric"][0]
        val_mae  = result.history["val_mae_diagonal_metric"][0]

        for k, v in zip(history2.keys(), [loss, val_loss, ssim, val_ssim, mae, val_mae]):
            history2[k].append(v)

        print(f"Época {epoch+1}/{epochs2} — loss:{loss:.4f} val_loss:{val_loss:.4f} "
              f"SSIM:{ssim:.4f} val_SSIM:{val_ssim:.4f}")

        if val_loss < best_loss2:
            best_loss2 = val_loss
            best_ssim2 = val_ssim
            if not smoke_test:
                autoencoder.save_weights(str(CHECKPOINTS / "phase2_best.weights.h5"))
                print(f"  ✅ Nuevo mejor Loss val: {best_loss2:.4f} — checkpoint guardado")
            else:
                print(f"  ✅ Nuevo mejor Loss val: {best_loss2:.4f} (smoke — no se guarda)")
            patience_counter2 = 0
        else:
            patience_counter2 += 1
            if patience_counter2 >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping en época {epoch+1}")
                break

        if patience_counter2 > 0 and patience_counter2 % 3 == 0:
            old_lr = float(autoencoder.optimizer.learning_rate)
            new_lr = max(old_lr * 0.5, 1e-7)
            autoencoder.optimizer.learning_rate.assign(new_lr)
            print(f"  LR reducido: {old_lr:.2e} → {new_lr:.2e}")

        if not smoke_test:
            with writer2.as_default():
                tf.summary.scalar("loss",            loss,     step=epoch)
                tf.summary.scalar("val_loss",        val_loss, step=epoch)
                tf.summary.scalar("ssim_metric",     ssim,     step=epoch)
                tf.summary.scalar("val_ssim_metric", val_ssim, step=epoch)
                tf.summary.scalar("mae_metric",      mae,      step=epoch)
                tf.summary.scalar("val_mae_metric",  val_mae,  step=epoch)
                tf.summary.scalar("lr", float(autoencoder.optimizer.learning_rate), step=epoch)
            writer2.flush()
    
    if not smoke_test:
        os.makedirs(str(LOGS), exist_ok=True)
        with open(str(LOGS / "history.json"), "w") as f:
            json.dump({
                "phase1": history1,
                "phase2": history2,
            }, f, indent=2)
        print("  Historial guardado en outputs/logs/history.json")
    else:
        print("  ⚠️  Smoke test completado — sin checkpoints ni logs guardados")

    return {"phase1": history1, "phase2": history2}



# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("GPU disponible:", tf.config.list_physical_devices("GPU"))

    smoke = "--smoke" in sys.argv

    if smoke:
        print("Modo smoke test — datos sintéticos, 2 épocas")
        # Para smoke test usamos la función simple con datos sintéticos
        from training.train import train as _train_orig
        train_ds = make_synthetic_dataset(256, BATCH_SIZE)
        val_ds   = make_synthetic_dataset(64,  BATCH_SIZE)
        # Smoke test usa la versión anterior inline
        autoencoder, encoder, decoder, emb_model = build_autoencoder(encoder_trainable=False)
        compile_model(autoencoder, lr=LR_PHASE1)
        autoencoder.fit(train_ds, validation_data=val_ds, epochs=2, verbose=1)
        print("✅ Smoke test completado")
    else:
        print(f"Entrenamiento real - BATCH_SIZE {BATCH_SIZE}")
        import pandas as pd
        from data.loader import load_and_prepare, COL_FECHA

        train_df, val_df, test_df, local_map, product_map, valid_combos = load_and_prepare()
        df_full   = pd.concat([train_df, val_df, test_df])
        train_end = train_df[COL_FECHA].max()
        val_end   = val_df[COL_FECHA].max()

        histories = train(
            df_full, local_map, product_map, valid_combos,
            train_end, val_end, smoke_test=False,
        )

        print("\n✅ Entrenamiento completado")
        print(f"   SSIM val Fase 1: {max(histories['phase1']['val_ssim_metric']):.4f}")
        print(f"   SSIM val Fase 2: {max(histories['phase2']['val_ssim_metric']):.4f}")