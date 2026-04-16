import sys
sys.path.append('/app')
import numpy as np
import pandas as pd
from data.loader import load_and_prepare, COL_FECHA, COL_LOCAL, COL_PRODUCTO, COL_UNIDADES, COL_STOCK
from config import WINDOW_SIZE, FORECAST_HORIZON, INPUT_STOCK_CERO_UMBRAL

train_df, val_df, test_df, local_map, product_map, valid_combos = load_and_prepare()
df_full   = pd.concat([train_df, val_df, test_df])
train_end = train_df[COL_FECHA].max()

df = df_full.copy()
df["local_idx"]    = df[COL_LOCAL].map(local_map).fillna(-1).astype(int)
df["producto_idx"] = df[COL_PRODUCTO].map(product_map).fillna(-1).astype(int)
valid_df = valid_combos[[COL_LOCAL, COL_PRODUCTO]].copy()
valid_df["_valid"] = True
df = df.merge(valid_df, on=[COL_LOCAL, COL_PRODUCTO], how="inner").drop(columns="_valid")

total = WINDOW_SIZE + FORECAST_HORIZON
n_quiebre = 0
n_normal  = 0
n_filtradas = 0

for (local, producto), group in df.groupby([COL_LOCAL, COL_PRODUCTO], sort=False):
    group = group.sort_values(COL_FECHA)
    n = len(group)
    if n < total:
        continue
    stock_arr  = group[COL_STOCK].values
    fechas_arr = group[COL_FECHA].values

    for start in range(0, n - total + 1):
        end_input  = start + WINDOW_SIZE
        end_target = end_input + FORECAST_HORIZON
        start_date = fechas_arr[start]

        if start_date >= train_end:
            continue

        frac_stock_cero = (stock_arr[start:end_input] == 0).mean()
        if frac_stock_cero > INPUT_STOCK_CERO_UMBRAL:
            n_filtradas += 1
            continue

        if (stock_arr[end_input:end_target] == 0).any():
            n_quiebre += 1
        else:
            n_normal += 1

total_v = n_quiebre + n_normal
print(f"\nVentanas train (todas las combinaciones):")
print(f"  Total válidas:  {total_v:,}")
print(f"  Con quiebre:    {n_quiebre:,} ({100*n_quiebre/total_v:.1f}%)")
print(f"  Sin quiebre:    {n_normal:,} ({100*n_normal/total_v:.1f}%)")
print(f"  Filtradas:      {n_filtradas:,}")
print(f"\n  Con QUIEBRE_RATIO=0.40:")
print(f"    Quiebres a usar por época: min({n_quiebre:,}, 40% del total)")
n_q_epoca = min(n_quiebre, int(total_v * 0.40))
n_n_epoca  = total_v - n_q_epoca
print(f"    → {n_q_epoca:,} quiebres + {n_n_epoca:,} normales")
