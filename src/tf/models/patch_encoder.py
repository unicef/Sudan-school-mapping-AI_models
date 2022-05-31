import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class PatchEncoder(layers.Layer):
    """
    An implementation of the encoder for the ViT model image patches.
    """

    def __init__(self, num_patches: int, projection_dim: int):
        """
        :param num_patches: the number of image patches.
        :param projection_dim: dimension of the dense embedding.
        """
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )


    def call(self, patch: np.array) -> layers.Layer:
        """
        Encodes the image patches and returns the embedding layer.
        :param patch: a list of image patches.
        :return: the embedding layer
        """
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded