import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import sys
sys.path.insert(0, '/app')

from config import (
    WINDOW_SIZE, FORECAST_HORIZON, SEED,
    INPUT_STOCK_CERO_UMBRAL, QUIEBRE_RATIO,
)
from gaf.transform import build_gaf_image
from data.loader import (
    COL_FECHA, COL_LOCAL, COL_PRODUCTO,
    COL_UNIDADES, COL_STOCK,
    load_and_prepare,
)


def build_tf_dataset(
    df_full: pd.DataFrame,
    local_map: dict,
    product_map: dict,
    valid_combos: pd.DataFrame,
    split: str = "train",
    train_end_date=None,
    val_end_date=None,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = SEED,
    quiebre_ratio: float = QUIEBRE_RATIO,
) -> tf.data.Dataset:
    """
    Construye un tf.data.Dataset lazy usando from_generator.
    Los GAF se generan on-the-fly por batch — nunca se carga todo en RAM.

    Cambios respecto a la versión anterior:
    - Filtro de input basado en stock=0 (reemplaza el filtro por ventas=0).
      Una semana sin ventas puede ser normal; stock=0 indica quiebre real.
    - En split='train', las ventanas se dividen en dos pools:
        · pool_quiebre: target tiene al menos 1 día con stock_cierre=0
        · pool_normal:  target sin quiebres de stock
      Se samplea quiebre_ratio del total desde pool_quiebre y el resto
      desde pool_normal, garantizando representación balanceada por época.
    - Val y test NO se balancean: se usan todas las ventanas disponibles
      para reflejar la distribución real.
    """
    df = df_full.copy()
    df["local_idx"]    = df[COL_LOCAL].map(local_map).fillna(-1).astype(int)
    df["producto_idx"] = df[COL_PRODUCTO].map(product_map).fillna(-1).astype(int)

    # Filtrar por combinaciones válidas (vectorizado)
    valid_df = valid_combos[[COL_LOCAL, COL_PRODUCTO]].copy()
    valid_df["_valid"] = True
    df = df.merge(valid_df, on=[COL_LOCAL, COL_PRODUCTO], how="inner").drop(columns="_valid")

    # ── Indexación de ventanas ────────────────────────────────────────────────
    # Cada entrada: (local, producto, start, tiene_quiebre)
    # tiene_quiebre = True si algún día del target tiene stock_cierre == 0
    window_index = []
    total = WINDOW_SIZE + FORECAST_HORIZON

    for (local, producto), group in df.groupby([COL_LOCAL, COL_PRODUCTO], sort=False):
        group = group.sort_values(COL_FECHA)
        n = len(group)

        if n < total:
            continue

        fechas_arr = group[COL_FECHA].values
        stock_arr  = group[COL_STOCK].values      # valores originales (pre log1p) para filtros
        ventas_arr = group[COL_UNIDADES].values   # solo para referencia, no se filtra aquí

        for start in range(0, n - total + 1):
            end_input  = start + WINDOW_SIZE
            end_target = end_input + FORECAST_HORIZON
            start_date = fechas_arr[start]

            # ── Filtro de input: descarta combinaciones con stock casi siempre en cero ──
            # Solo en train; en val/test evaluamos todo para medir distribución real.
            if split == "train":
                frac_stock_cero_input = (stock_arr[start:end_input] == 0).mean()
                if frac_stock_cero_input > INPUT_STOCK_CERO_UMBRAL:
                    continue

            # ── Filtro de split por fecha ─────────────────────────────────────
            if split == "train" and train_end_date is not None:
                if start_date >= train_end_date:
                    continue
            elif split == "val" and train_end_date is not None and val_end_date is not None:
                if start_date < train_end_date or start_date >= val_end_date:
                    continue
            elif split == "test" and val_end_date is not None:
                if start_date < val_end_date:
                    continue

            # ── Etiqueta de quiebre en el target ─────────────────────────────
            # Quiebre = al menos 1 día con stock_cierre == 0 en los 7 días target.
            # Usamos stock_arr original (pre log1p) para comparar con 0 directamente.
            tiene_quiebre = bool((stock_arr[end_input:end_target] == 0).any())

            window_index.append((local, producto, start, tiene_quiebre))

    if not window_index:
        raise ValueError(f"No se generaron ventanas para split='{split}'.")

    # ── Balanceo por época (solo train) ──────────────────────────────────────
    if split == "train":
        pool_quiebre = [(l, p, s) for l, p, s, q in window_index if q]
        pool_normal  = [(l, p, s) for l, p, s, q in window_index if not q]

        rng = np.random.default_rng(seed)

        if len(pool_quiebre) == 0:
            # Sin quiebres disponibles: usar todo el pool normal sin balanceo
            final_index = pool_normal
            print(f"  ⚠️  Sin ventanas de quiebre en train — se usa pool normal completo")
        else:
            # El pool de quiebres define el tamaño del dataset balanceado.
            # n_normal se calcula para que quiebres representen exactamente quiebre_ratio.
            # Ejemplo: 65k quiebres con ratio=0.40 → 97k normales → 163k total (40/60)
            n_quiebre = len(pool_quiebre)
            n_normal  = int(n_quiebre * (1.0 - quiebre_ratio) / quiebre_ratio)
            n_normal  = min(n_normal, len(pool_normal))  # no pedir más de los disponibles

            sampled_q = pool_quiebre  # usamos todos los quiebres siempre
            idx_n     = rng.choice(len(pool_normal), n_normal, replace=False)
            sampled_n = [pool_normal[i] for i in idx_n]

            final_index = sampled_q + sampled_n
            rng.shuffle(final_index)

            print(f"  Ventanas indexadas ({split}): {len(window_index):,} total "
                  f"→ {len(pool_quiebre):,} quiebres ({100*len(pool_quiebre)/len(window_index):.1f}%) "
                  f"/ {len(pool_normal):,} normales")
            print(f"  Balanceo aplicado (ratio={quiebre_ratio:.0%}): "
                  f"{len(sampled_q):,} quiebres + {len(sampled_n):,} normales "
                  f"= {len(final_index):,} ventanas por época")

    else:
        # Val y test: todas las ventanas, sin balanceo, orden original
        final_index = [(l, p, s) for l, p, s, q in window_index]
        n_quiebres_val = sum(1 for _, _, _, q in window_index if q)
        print(f"  Ventanas indexadas ({split}): {len(final_index):,} "
              f"({n_quiebres_val:,} con quiebre de stock, "
              f"{len(final_index)-n_quiebres_val:,} sin quiebre)")

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(final_index)

    # ── Datos por grupo para acceso rápido en el generator ───────────────────
    group_data = {}
    for (local, producto), grp in df.groupby([COL_LOCAL, COL_PRODUCTO], sort=False):
        grp = grp.sort_values(COL_FECHA).reset_index(drop=True)
        group_data[(local, producto)] = {
            "unidades":    np.log1p(grp[COL_UNIDADES].values.astype(np.float32)),
            "stock":       np.log1p(grp[COL_STOCK].values.astype(np.float32)),
            "precio":      grp["precio"].values.astype(np.float32),
            "local_idx":   np.int32(grp["local_idx"].iloc[0]),
            "product_idx": np.int32(grp["producto_idx"].iloc[0]),
        }

    # ── Generator (sin cambios respecto a versión anterior) ───────────────────
    def generator():
        for local, producto, start in final_index:
            g = group_data[(local, producto)]
            end_input  = start + WINDOW_SIZE
            end_target = end_input + FORECAST_HORIZON

            gaf_input = build_gaf_image(
                g["unidades"][start:end_input],
                g["stock"][start:end_input],
                g["precio"][start:end_input],
            )
            gaf_target = build_gaf_image(
                g["unidades"][end_input:end_target],
                g["stock"][end_input:end_target],
                g["precio"][end_input:end_target],
            )

            yield (
                {
                    "gaf_input":    gaf_input.astype(np.float32),
                    "local_idx":    g["local_idx"],
                    "producto_idx": g["product_idx"],
                },
                gaf_target.astype(np.float32),
            )

    output_signature = (
        {
            "gaf_input":    tf.TensorSpec(shape=(WINDOW_SIZE, WINDOW_SIZE, 3), dtype=tf.float32),
            "local_idx":    tf.TensorSpec(shape=(),                            dtype=tf.int32),
            "producto_idx": tf.TensorSpec(shape=(),                            dtype=tf.int32),
        },
        tf.TensorSpec(shape=(FORECAST_HORIZON, FORECAST_HORIZON, 3), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_tf_dataset_sampled(
    df_full: pd.DataFrame,
    local_map: dict,
    product_map: dict,
    valid_combos: pd.DataFrame,
    split: str = "train",
    train_end_date=None,
    val_end_date=None,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = SEED,
    n_combos: int = 1000,
    quiebre_ratio: float = QUIEBRE_RATIO,
) -> tf.data.Dataset:
    """
    Samplea n_combos combinaciones y construye un dataset lazy con balanceo.

    El parámetro quiebre_ratio se pasa a build_tf_dataset y solo tiene
    efecto en split='train'. Val y test ignoran el balanceo.
    """
    if n_combos < len(valid_combos):
        sampled = valid_combos.sample(n=n_combos, random_state=seed)
    else:
        sampled = valid_combos

    return build_tf_dataset(
        df_full=df_full,
        local_map=local_map,
        product_map=product_map,
        valid_combos=sampled,
        split=split,
        train_end_date=train_end_date,
        val_end_date=val_end_date,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        quiebre_ratio=quiebre_ratio,
    )


if __name__ == "__main__":
    from config import BATCH_SIZE

    print("Cargando datos...")
    train, val, test, local_map, product_map, valid_combos = load_and_prepare()

    df_full   = pd.concat([train, val, test])
    train_end = train[COL_FECHA].max()
    val_end   = val[COL_FECHA].max()

    print("\nSmoke test — dataset train (50 combinaciones)...")
    valid_sample = valid_combos.head(50)

    ds = build_tf_dataset_sampled(
        df_full, local_map, product_map, valid_sample,
        split="train",
        train_end_date=train_end,
        val_end_date=val_end,
        batch_size=BATCH_SIZE,
        shuffle=True,
        n_combos=50,
    )

    for inputs, gaf_tgt in ds.take(1):
        print(f"\nShapes del batch:")
        print(f"  gaf_input:    {inputs['gaf_input'].shape}")
        print(f"  local_idx:    {inputs['local_idx'].shape}")
        print(f"  producto_idx: {inputs['producto_idx'].shape}")
        print(f"  gaf_target:   {gaf_tgt.shape}")