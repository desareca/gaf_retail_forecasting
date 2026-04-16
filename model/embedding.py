# model/embedding.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    NUM_LOCALES, NUM_PRODUCTOS,
    LOCAL_EMB_DIM, PRODUCTO_EMB_DIM,
    EMB_MLP_HIDDEN, EMB_OUTPUT_DIM, EMB_DROPOUT,
)


def build_embedding_model(
    num_locales: int = NUM_LOCALES,
    num_productos: int = NUM_PRODUCTOS,
    local_dim: int = LOCAL_EMB_DIM,
    producto_dim: int = PRODUCTO_EMB_DIM,
    mlp_hidden: int = EMB_MLP_HIDDEN,
    output_dim: int = EMB_OUTPUT_DIM,
    dropout: float = EMB_DROPOUT,
) -> keras.Model:
    """
    Modelo de embedding que combina identidad de local y producto
    en un vector condicionante para FiLM.

    Inputs:
        local_idx    : (batch,)  — índice entero del local
        producto_idx : (batch,)  — índice entero del producto

    Output:
        embedding    : (batch, output_dim)  — vector condicionante
    """
    local_input = keras.Input(shape=(), dtype=tf.int32, name="local_idx")
    producto_input = keras.Input(shape=(), dtype=tf.int32, name="producto_idx")

    # Embeddings base
    local_emb = layers.Embedding(
        input_dim=num_locales,
        output_dim=local_dim,
        name="local_embedding",
    )(local_input)  # (batch, local_dim)

    producto_emb = layers.Embedding(
        input_dim=num_productos,
        output_dim=producto_dim,
        name="producto_embedding",
    )(producto_input)  # (batch, producto_dim)

    # Concatenar: (batch, local_dim + producto_dim)
    x = layers.Concatenate(name="concat_embeddings")([local_emb, producto_emb])

    # MLP: [local_dim + producto_dim] → [mlp_hidden] → [output_dim]
    x = layers.Dense(mlp_hidden, activation="relu", name="mlp_hidden")(x)
    x = layers.Dropout(dropout, name="mlp_dropout")(x)
    x = layers.Dense(output_dim, activation="relu", name="mlp_output")(x)

    return keras.Model(
        inputs=[local_input, producto_input],
        outputs=x,
        name="embedding_model",
    )