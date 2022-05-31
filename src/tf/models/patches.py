import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Patches(layers.Layer):
    """
    A layer that creates image patches for the ViT model.
    """

    def __init__(self, patch_size: int):
        super(Patches, self).__init__()
        self.patch_size = patch_size


    def call(self, images: np.array) -> np.array:
        """
        Creates and returns image patches for the ViT model
        :param images: an images array
        :return: images patches
        """
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

        return patches
    