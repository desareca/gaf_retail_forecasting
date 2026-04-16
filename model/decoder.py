# model/decoder.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotNormal
from model.film import FiLMBlock
from config import FORECAST_HORIZON


def _upblock(x, embedding, num_channels, film_name, block_name):
    """
    Bloque de upsampling:
        Upsample bilinear → Conv 3×3 → BN → ReLU → FiLM
    """
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"{block_name}_upsample")(x)
    x = layers.Conv2D(num_channels, 3, padding="same", use_bias=False, name=f"{block_name}_conv")(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn")(x)
    x = layers.ReLU(name=f"{block_name}_relu")(x)
    x = FiLMBlock(num_channels=num_channels, name=film_name)(x, embedding)
    return x


def build_decoder(
    bottleneck_shape: tuple = (3, 3, 1280),
    embedding_dim: int = 64,
    output_shape: tuple = (7, 7, 3),
) -> keras.Model:
    """
    Decoder que reconstruye una imagen GAF 7×7×3 desde el bottleneck del encoder.

    Arquitectura:
        (3×3×1280) → UpBlock1 → (6×6×256) → FiLM
                   → UpBlock2 → (12×12×128) — recortamos a (7×7×128) → FiLM
                   → Conv 1×1 → (7×7×3) + tanh

    Nota: el upsampling bilineal lleva 3×3 → 6×6 → 12×12.
    Recortamos 12×12 a 7×7 con un crop central antes de la conv final.

    Inputs:
        bottleneck : (batch, 3, 3, 1280)
        embedding  : (batch, 64)

    Output:
        (batch, 7, 7, 3)  — imagen GAF predicha, valores en [-1, 1]
    """
    bottleneck_input = keras.Input(shape=bottleneck_shape, name="bottleneck_input")
    embedding_input  = keras.Input(shape=(embedding_dim,),  name="embedding_input")

    x = bottleneck_input

    # ── Bloque 1: (3×3×1280) → (6×6×256) ──────────────────────────────────
    x = _upblock(x, embedding_input, num_channels=256,
                 film_name="film_block1", block_name="upblock1")

    # ── Bloque 2: (6×6×256) → (12×12×128) ─────────────────────────────────
    x = _upblock(x, embedding_input, num_channels=128,
                 film_name="film_block2", block_name="upblock2")

    # ── Crop central 12×12 → 7×7 ───────────────────────────────────────────
    # (12 - 7) // 2 = 2 → recortamos 2 píxeles de cada lado
    #x = layers.Cropping2D(cropping=((2, 3), (2, 3)), name=f"crop_to_{FORECAST_HORIZON}x{FORECAST_HORIZON}")(x)
    x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=1, padding='valid', name=f"up_to_{FORECAST_HORIZON}x{FORECAST_HORIZON}")(x)

    # resultado: (7×7×128)

    # ── Conv final → (7×7×3) + tanh ────────────────────────────────────────
    # x = layers.Conv2D(3, 1, padding="same", activation="tanh", name="output_conv")(x)
    # x = layers.Conv2D(3, 1, padding="same", activation="softsign", kernel_initializer=GlorotNormal(), name="output_conv")(x)
    x = layers.Conv2D(3, 1, padding="same", kernel_initializer=GlorotNormal(), name="output_conv_linear")(x)
    # x = layers.BatchNormalization(momentum=0.98, name="bn_output")(x)
    # x = layers.LayerNormalization(axis=(1, 2), name="ln_output")(x) # instance_norm_emulated
    x = layers.Activation('softsign', name="output_conv")(x)
    # x = layers.Activation('tanh', name="output_conv")(x)

    return keras.Model(
        inputs=[bottleneck_input, embedding_input],
        outputs=x,
        name="decoder",
    )