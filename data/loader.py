import pandas as pd
import numpy as np
from pathlib import Path
import json

import sys
sys.path.insert(0, '/app')
from config import (
    DATA_RAW, MAPPINGS,
    TRAIN_RATIO, VAL_RATIO,
    IMPUTE_WINDOW_DAYS, PRICE_FFILL_DAYS,
    WINDOW_SIZE, FORECAST_HORIZON,
)

import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


CSV_FILENAME = "ventas_diarias.csv"

COL_FECHA    = "date"
COL_LOCAL    = "store_id"
COL_PRODUCTO = "retailer_sku"
COL_UNIDADES = "units"
COL_MONTO    = "revenue"
COL_STOCK    = "stock"


def load_raw(filepath: Path = None) -> pd.DataFrame:
    if filepath is None:
        filepath = DATA_RAW / CSV_FILENAME
    df = pd.read_csv(filepath, sep=";", parse_dates=[COL_FECHA], low_memory=False)
    df = df[[COL_FECHA, COL_LOCAL, COL_PRODUCTO,
             COL_UNIDADES, COL_MONTO, COL_STOCK]].copy()
    df = df.sort_values([COL_LOCAL, COL_PRODUCTO, COL_FECHA]).reset_index(drop=True)
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[COL_UNIDADES] = df[COL_UNIDADES].clip(lower=0)
    df[COL_MONTO]    = df[COL_MONTO].clip(lower=0)
    df[COL_STOCK]    = df[COL_STOCK].clip(lower=0)

    def impute_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()

        mask_u = g[COL_UNIDADES].isna()
        if mask_u.any():
            rolling_u = g[COL_UNIDADES].fillna(0).rolling(IMPUTE_WINDOW_DAYS, min_periods=1).mean()
            g.loc[mask_u, COL_UNIDADES] = rolling_u[mask_u]

        mask_m = g[COL_MONTO].isna()
        if mask_m.any():
            rolling_m = g[COL_MONTO].fillna(0).rolling(IMPUTE_WINDOW_DAYS, min_periods=1).mean()
            g.loc[mask_m, COL_MONTO] = rolling_m[mask_m]

        for idx in g.index[g[COL_STOCK].isna()]:
            pos = g.index.get_loc(idx)
            if pos > 0:
                prev_idx = g.index[pos - 1]
                prev_stock = g.loc[prev_idx, COL_STOCK]
                curr_units = g.loc[idx, COL_UNIDADES]
                if pd.notna(prev_stock):
                    g.loc[idx, COL_STOCK] = max(0.0, prev_stock - curr_units)
                else:
                    g.loc[idx, COL_STOCK] = 0.0
            else:
                g.loc[idx, COL_STOCK] = 0.0

        return g

    df = df.groupby([COL_LOCAL, COL_PRODUCTO], group_keys=False).apply(impute_group)

    df["precio"] = np.where(
        df[COL_UNIDADES] > 0,
        df[COL_MONTO] / df[COL_UNIDADES],
        np.nan
    )

    def impute_precio(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["precio"] = g["precio"].ffill(limit=PRICE_FFILL_DAYS)
        mediana = g["precio"].median()
        g["precio"] = g["precio"].fillna(mediana if pd.notna(mediana) else 0.0)
        return g

    df = df.groupby([COL_LOCAL, COL_PRODUCTO], group_keys=False).apply(impute_precio)

    return df.reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple:
    fechas = sorted(df[COL_FECHA].unique())
    n = len(fechas)

    train_end = fechas[int(n * TRAIN_RATIO) - 1]
    val_end   = fechas[int(n * VAL_RATIO) - 1]

    train = df[df[COL_FECHA] <= train_end].copy()
    val   = df[(df[COL_FECHA] > train_end) & (df[COL_FECHA] <= val_end)].copy()
    test  = df[df[COL_FECHA] > val_end].copy()

    return train, val, test


def build_mappings(df: pd.DataFrame) -> tuple:
    locales   = sorted(df[COL_LOCAL].unique())
    productos = sorted(df[COL_PRODUCTO].unique())

    local_map   = {v: i for i, v in enumerate(locales)}
    product_map = {v: i for i, v in enumerate(productos)}

    MAPPINGS.mkdir(parents=True, exist_ok=True)
    with open(MAPPINGS / "local_map.json", "w") as f:
        json.dump({str(k): v for k, v in local_map.items()}, f, indent=2)
    with open(MAPPINGS / "product_map.json", "w") as f:
        json.dump({str(k): v for k, v in product_map.items()}, f, indent=2)

    return local_map, product_map


def get_valid_combinations(df: pd.DataFrame,
                           min_days: int = WINDOW_SIZE + FORECAST_HORIZON) -> pd.DataFrame:
    counts = df.groupby([COL_LOCAL, COL_PRODUCTO])[COL_FECHA].nunique()
    valid  = counts[counts >= min_days].reset_index()
    valid.columns = [COL_LOCAL, COL_PRODUCTO, "n_dias"]
    return valid


def load_and_prepare(filepath: Path = None):
    print("Cargando datos...")
    df = load_raw(filepath)
    print(f"  Raw: {len(df):,} filas")

    print("Imputando...")
    df = impute(df)
    print(f"  Después de imputación: {len(df):,} filas")

    print("Split temporal...")
    train, val, test = temporal_split(df)
    print(f"  Train: {len(train):,} filas | {train[COL_FECHA].min().date()} → {train[COL_FECHA].max().date()}")
    print(f"  Val:   {len(val):,} filas | {val[COL_FECHA].min().date()} → {val[COL_FECHA].max().date()}")
    print(f"  Test:  {len(test):,} filas | {test[COL_FECHA].min().date()} → {test[COL_FECHA].max().date()}")

    print("Construyendo mappings...")
    local_map, product_map = build_mappings(df)
    print(f"  Locales:   {len(local_map)}")
    print(f"  Productos: {len(product_map)}")

    print("Filtrando combinaciones válidas...")
    valid_combos = get_valid_combinations(df)
    print(f"  Combinaciones con >= {WINDOW_SIZE + FORECAST_HORIZON} días: {len(valid_combos):,}")

    return train, val, test, local_map, product_map, valid_combos


if __name__ == "__main__":
    load_and_prepare()