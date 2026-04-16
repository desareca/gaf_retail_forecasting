# model/encoder.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


def build_encoder(input_shape: tuple = (90, 90, 3), trainable: bool = False) -> keras.Model:
    """
    Encoder basado en EfficientNetB0 pre-entrenada en ImageNet.

    - En Fase 1 de entrenamiento: trainable=False (encoder congelado)
    - En Fase 2: se descongela desde afuera pasando trainable=True
      o llamando a unfreeze_top_blocks()

    Input:  (batch, 90, 90, 3)
    Output: (batch, 3, 3, 1280)  — bottleneck de EfficientNetB0
    """
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = trainable

    inputs = keras.Input(shape=input_shape, name="gaf_input")

    # EfficientNetB0 espera píxeles en [0, 255] o preprocesados.
    # Nuestras imágenes GAF salen de tanh → [-1, 1].
    # Reescalamos a [0, 255] para compatibilidad con los pesos ImageNet.
    x = (inputs + 1.0) * 127.5  # [-1,1] → [0,255]

    x = base(x, training=False)  # training=False mantiene BatchNorm en modo inference

    return keras.Model(inputs=inputs, outputs=x, name="encoder")


def unfreeze_top_blocks(encoder: keras.Model, num_blocks: int = 3) -> None:
    """
    Descongela los últimos num_blocks bloques del encoder para fine-tuning (Fase 2).
    Modifica el modelo in-place.
    """
    base = encoder.layers[3]  # EfficientNetB0 es la 4ta capa (después de Input, rescale, call)
    base.trainable = True

    # Congelar todo excepto los últimos num_blocks bloques
    block_names = [l.name for l in base.layers if l.name.startswith("block")]
    unique_blocks = sorted(set(n.split("_")[0] + "_" + n.split("_")[1] for n in block_names))
    blocks_to_freeze = unique_blocks[:-num_blocks]

    for layer in base.layers:
        layer.trainable = True
        for block_prefix in blocks_to_freeze:
            if layer.name.startswith(block_prefix):
                layer.trainable = False
                break

    frozen = sum(1 for l in base.layers if not l.trainable)
    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"Encoder — capas congeladas: {frozen}, entrenables: {trainable}")