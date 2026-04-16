# model/autoencoder.py
import tensorflow as tf
from tensorflow import keras
from model.encoder import build_encoder
from model.decoder import build_decoder
from model.embedding import build_embedding_model
from config import EMB_OUTPUT_DIM, IMAGE_SIZE


def build_autoencoder(
    gaf_input_shape: tuple = (90, 90, 3),
    gaf_output_shape: tuple = (7, 7, 3),
    embedding_dim: int = EMB_OUTPUT_DIM,
    encoder_trainable: bool = False,
) -> tuple[keras.Model, keras.Model, keras.Model, keras.Model]:
    """
    Ensambla encoder + embedding + decoder en un modelo end-to-end.

    Inputs del autoencoder:
        gaf_input    : (batch, 90, 90, 3)
        local_idx    : (batch,)
        producto_idx : (batch,)

    Output:
        gaf_pred     : (batch, 7, 7, 3)

    Returns:
        autoencoder, encoder, decoder, embedding_model
    """
    # ── Submodelos ──────────────────────────────────────────────────────────
    encoder         = build_encoder(input_shape=gaf_input_shape, trainable=encoder_trainable)
    embedding_model = build_embedding_model()
    decoder         = build_decoder(
        bottleneck_shape=encoder.output_shape[1:],  # (3, 3, 1280)
        embedding_dim=embedding_dim,
        output_shape=gaf_output_shape,
    )

    # ── Inputs ──────────────────────────────────────────────────────────────
    gaf_input    = keras.Input(shape=gaf_input_shape, name="gaf_input",    dtype=tf.float32)
    local_idx    = keras.Input(shape=(),              name="local_idx",     dtype=tf.int32)
    producto_idx = keras.Input(shape=(),              name="producto_idx",  dtype=tf.int32)

    # ── Forward pass ────────────────────────────────────────────────────────
    bottleneck = encoder(gaf_input, training=False)
    embedding  = embedding_model([local_idx, producto_idx], training=True)
    gaf_pred   = decoder([bottleneck, embedding], training=True)

    autoencoder = keras.Model(
        inputs=[gaf_input, local_idx, producto_idx],
        outputs=gaf_pred,
        name="gaf_autoencoder",
    )

    return autoencoder, encoder, decoder, embedding_model