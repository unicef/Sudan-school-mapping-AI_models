import tensorflow as tf
from tensorflow.keras.models import Model
from src.tf.models.resnet import ResNetMulticlassClassifier
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout


class ResNet101(ResNetMulticlassClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def build(self):
        resnet_model = tf.keras.applications.ResNet50(
            weights=None,
            include_top=False,
            pooling=False,
            input_tensor=Input(shape=self.input_shape)
        )
        x = AveragePooling2D(pool_size=(7, 7))(resnet_model.output)
        x = Flatten(name="flatten")(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(self.classes_num, activation="softmax")(x)
        self.model = Model(inputs=resnet_model.input, outputs=x)
        self.model.summary()


if __name__ == "__main__":
    resnet = ResNet101(classes_num=2)
    resnet.build()
    resnet.train(images_dir=r"D:\op\datasets\v5_structured")