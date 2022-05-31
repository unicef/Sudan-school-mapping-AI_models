import tensorflow as tf
from tensorflow.keras.models import Model
from src.tf.models.resnet import ResNetMulticlassClassifier
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, \
    AveragePooling2D, Flatten, ZeroPadding2D, Add, MaxPool2D, Dense, Dropout


"""
The customized ResNet implementation found in the demo:
https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
"""
class ResNet18(ResNetMulticlassClassifier):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_identity_block(self, x, filter, kernel=(3, 3)):
        x2 = Conv2D(filter, kernel, padding="same")(x)
        x2 = BatchNormalization(axis=3)(x2)
        x2 = Activation("relu")(x2)
        x2 = Conv2D(filter, kernel, padding="same")(x2)
        x2 = BatchNormalization(axis=3)(x2)
        x3 = Add()([x2, x])
        x3 = Activation("relu")(x3)

        return x3


    def get_conv_block(self, x, filter, kernel=(3, 3), strides=(2, 2)):
        x2 = Conv2D(filter, kernel, padding="same", strides=strides)(x)
        x2 = BatchNormalization(axis=3)(x2)
        x2 = Activation("relu")(x2)
        x2 = Conv2D(filter, kernel, padding="same")(x2)
        x2 = BatchNormalization(axis=3)(x2)
        x3 = Conv2D(filter, (1, 1), strides=strides)(x)
        x4 = Add()([x2, x3])
        x4 = Activation("relu")(x4)

        return x4


    def build(self):
        input = Input(self.input_shape)
        x = ZeroPadding2D((3, 3))(input)
        x = Conv2D(64, 7, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D(3, strides=2, padding="same")(x)
        block_layers = [3, 6]
        filter_size = 64

        for i in range(2):
            if i == 0:
                for j in range(block_layers[i]):
                    x = self.get_identity_block(x, filter=filter_size)
            else:
                filter_size = filter_size * 2
                x = self.get_conv_block(x, filter=filter_size)
                for j in range(block_layers[i] - 1):
                    x = self.get_identity_block(x, filter=filter_size)

        x = AveragePooling2D(pool_size=(2, 2), padding="same")(x)
        x = Flatten(name="flatten")(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(self.classes_num, activation="softmax")(x)
        self.model = Model(inputs=input, outputs=x, name='Resnet18')
        self.model.summary()


if __name__ == "__main__":
    resnet = ResNet18(classes_num=2)
    resnet.build()
    resnet.train(images_dir=r"D:\op\datasets\v5_structured")