# model/film.py
import tensorflow as tf
from tensorflow.keras import layers


class FiLMBlock(layers.Layer):
    """
    Feature-wise Linear Modulation (FiLM).
    Perez et al. 2018 — https://arxiv.org/abs/1709.07871

    Recibe un feature map y un vector condicionante (embedding),
    y aplica una modulación afín canal a canal:

        output = gamma * feature_map + beta

    donde gamma y beta son proyecciones lineales del embedding,
    con broadcast sobre las dimensiones espaciales H y W.

    El shape del feature map NO cambia.

    Args:
        num_channels : número de canales C del feature map que modulará este bloque.
        name         : nombre de la capa (útil para inspección del modelo).
    """

    def __init__(self, num_channels: int, name: str = "film_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_channels = num_channels

        # Cada bloque tiene sus propias proyecciones (Opción A del diseño).
        # El mismo vector embedding pasa por capas distintas en cada bloque del decoder.
        self.gamma_proj = layers.Dense(
            num_channels,
            activation=None,
            use_bias=True,
            bias_initializer='ones',
            name=f"{name}_gamma",
        )
        self.beta_proj = layers.Dense(
            num_channels,
            activation=None,
            use_bias=True,
            name=f"{name}_beta",
        )

    def call(self, feature_map: tf.Tensor, embedding: tf.Tensor) -> tf.Tensor:
        """
        Args:
            feature_map : (batch, H, W, C)
            embedding   : (batch, emb_dim)

        Returns:
            (batch, H, W, C)  — mismo shape que feature_map
        """
        # Proyectar embedding a gamma y beta: (batch, C)
        gamma = self.gamma_proj(embedding)  # (batch, C)
        beta = self.beta_proj(embedding)    # (batch, C)

        # Expandir para broadcast sobre H y W: (batch, 1, 1, C)
        gamma = tf.reshape(gamma, (-1, 1, 1, self.num_channels))
        beta = tf.reshape(beta, (-1, 1, 1, self.num_channels))

        # Modulación afín: misma operación para cada píxel (H, W)
        return gamma * feature_map + beta

    def get_config(self):
        config = super().get_config()
        config.update({"num_channels": self.num_channels})
        return config