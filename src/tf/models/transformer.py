import tensorflow as tf
from typing import Tuple, List
from tensorflow.keras import layers
from src.tf.models.common import BaseModel
from src.tf.models.patches import Patches
from src.tf.models.patch_encoder import PatchEncoder


class TransformerBinaryClassifier(BaseModel):
    """
    Binary classification model base on the Transformer architecture
    The script contains a customized implementation of the ViT model presented in the demo:
    https://keras.io/examples/vision/image_classification_with_vision_transformer/
    """

    def __init__(self, *args, **kwargs):
        self.num_classes = 2
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.batch_size = 128
        self.num_epochs = 50
        self.image_size = 256
        self.patch_size = 32
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection_dim = 256
        self.num_heads = 4
        self.transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]
        self.transformer_layers = 12
        self.mlp_head_units = [4096, 2048]
        super().__init__(*args, **kwargs)


    def mlp(self, x: layers.Layer, hidden_units: List[int], dropout_rate: float) -> layers.Layer:
        """
        Creates dense layers of the transfomer model

        :param x: the input data
        :param hidden_units: the number of units for the hidden layer
        :param dropout_rate: the dropout rate for the droput layer
        :return: the last layer of the model
        """
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x


    def create_vit_classifier(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """
        Creates and returns a vision transformer model

        :param input_shape: an image input shape
        :return: ViT model
        """
        inputs = layers.Input(shape=input_shape)
        # Augment data.
        # augmented = data_augmentation(inputs)
        # Create patches.
        # patches = Patches(patch_size)(augmented)
        # Encode patches.
        patches = Patches(self.patch_size)(inputs)
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = self.mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(self.num_classes)(features)
        # Create the Keras model.
        model = tf.keras.Model(inputs=inputs, outputs=logits)

        return model


    def build(self):
        model = self.create_vit_classifier(input_shape=(self.image_size, self.image_size, 3))
        return model


    def train(self):
        pass


    def get_predicted_probability(self, prediction_result: List[List]) -> float:
        probabilities = tf.keras.activations.softmax(
            tf.convert_to_tensor(prediction_result)).numpy()
        return probabilities[0][1]



