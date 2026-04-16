# gaf/transform.py

import numpy as np


def normalize_series(series: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Normaliza una serie al rango [-1, 1].
    Maneja series constantes devolviendo un array de ceros.

    Returns:
        (series_norm, min_val, max_val)
    """
    min_val = 0# series.min()
    max_val = series.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(series, dtype=np.float32), min_val, max_val

    norm = 2.0 * (series - min_val) / (max_val - min_val) - 1.0
    return norm.astype(np.float32), min_val, max_val


def build_gasf(series_norm: np.ndarray) -> np.ndarray:
    """
    Construye la matriz GASF completa.
    G[i,j] = cos(phi_i + phi_j)
    donde phi = arccos(x), x en [-1, 1]

    Shape: (N, N)
    """
    phi = np.arccos(np.clip(series_norm, -1.0, 1.0))
    gasf = np.cos(phi[:, None] + phi[None, :])
    return gasf.astype(np.float32)


def build_gadf(series_norm: np.ndarray) -> np.ndarray:
    """
    Construye la matriz GADF completa.
    G[i,j] = sin(phi_i - phi_j)
    donde phi = arccos(x), x en [-1, 1]

    Propiedad: diagonal siempre es 0, anti-simétrica.

    Shape: (N, N)
    """
    phi = np.arccos(np.clip(series_norm, -1.0, 1.0))
    gadf = np.sin(phi[:, None] - phi[None, :])
    return gadf.astype(np.float32)


def build_hybrid_channel(series_norm: np.ndarray) -> np.ndarray:
    """
    Construye una imagen híbrida GASF/GADF para una sola serie:

        Triángulo superior + diagonal: GASF
        Triángulo inferior:            GADF

    Esto aprovecha la simetría redundante de GASF llenando
    el triángulo inferior con información de cambio (GADF).
    No hay conflicto en la diagonal porque GADF[i,i] = sin(0) = 0,
    y la reemplazamos con GASF[i,i] que contiene el valor original.

    Shape: (N, N)
    """
    gasf = build_gasf(series_norm)
    gadf = build_gadf(series_norm)

    # Máscara del triángulo inferior (sin diagonal)
    n = len(series_norm)
    lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)

    hybrid = gasf.copy()
    hybrid[lower_mask] = gadf[lower_mask]

    return hybrid.astype(np.float32)


def build_gaf_image(
    ventas: np.ndarray,
    stock: np.ndarray,
    precio: np.ndarray,
) -> np.ndarray:
    """
    Construye imagen GAF híbrida de 3 canales (H, W, 3).

    Cada canal es una imagen híbrida GASF/GADF independiente:
        Canal 0 (R): ventas
        Canal 1 (G): stock
        Canal 2 (B): precio

    Internamente cada canal:
        - Triángulo superior + diagonal: GASF  (magnitud, correlación)
        - Triángulo inferior:            GADF  (cambios, transiciones)

    Args:
        ventas, stock, precio: series 1D de igual longitud N

    Returns:
        imagen float32 en [-1, 1], shape (N, N, 3)
    """
    assert len(ventas) == len(stock) == len(precio), \
        "Las tres series deben tener la misma longitud"

    ventas_norm, _, _ = normalize_series(ventas)
    stock_norm, _, _  = normalize_series(stock)
    precio_norm, _, _ = normalize_series(precio)

    r = build_hybrid_channel(ventas_norm)
    g = build_hybrid_channel(stock_norm)
    b = build_hybrid_channel(precio_norm)

    return np.stack([r, g, b], axis=-1)  # (N, N, 3)


def build_hybrid_channel(series_norm: np.ndarray) -> np.ndarray:
    gasf = build_gasf(series_norm)
    gadf = build_gadf(series_norm)

    n = len(series_norm)
    lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)

    hybrid = gasf.copy()
    hybrid[lower_mask] = gadf[lower_mask]

    # Sobrescribir la diagonal con la serie normalizada directamente
    # Esto permite recuperarla sin ambigüedad
    np.fill_diagonal(hybrid, series_norm)

    return hybrid.astype(np.float32)


def invert_diagonal(channel: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    x_norm = np.diag(channel)  # ya es la serie normalizada, sin inversión matemática
    if max_val - min_val < 1e-8:
        return np.full(len(x_norm), min_val, dtype=np.float32)
    return ((x_norm + 1.0) / 2.0 * (max_val - min_val) + min_val).astype(np.float32)