import dill
import logging
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from src.tf.models.loss_utils import categorical_focal_loss
from src.tf.models.object_classifier import ObjectClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.tf.models.loss_learning_rate_scheduler import LossLearningRateScheduler
from src.img_utils import get_balanced_data_generators, get_images_per_category, get_categories_weights


class ResNetMulticlassClassifier(ObjectClassifier):

    def __init__(self, *args, **kwargs):
        # Number of different classes of images
        self.classes_num = 1 if "classes_num" not in kwargs else kwargs["classes_num"]
        # Weights assigned to different classes of images
        self.classes_weights = [1, 1] if "classes_weights" not in kwargs else kwargs["classes_weights"]
        dill_dumps = dill.dumps(
            categorical_focal_loss(gamma=2., alpha=[self.classes_weights]))
        kwargs["custom_objects"] = {"focal_loss": dill.loads(dill_dumps)}
        super().__init__(*args, **kwargs)


    def train(self, *args, **kwargs):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        images_dir = kwargs["images_dir"]
        best_val_loss_save_path = super().get_model_save_path(prefix="resnet_best_val_loss")
        model_save_path = super().get_model_save_path(prefix="resnet_model")
        learning_rate = 1e-4 if "lr" not in kwargs else float(kwargs["lr"])
        epochs = 1 if "epochs" not in kwargs else kwargs["epochs"]
        batch_size = 32 if "batch_size" not in kwargs else kwargs["batch_size"]
        categories, total_images = get_images_per_category(images_dir)
        categories_weight = get_categories_weights(categories, total_images)
        loss_weights = [categories_weight[category] for category, index in categories_weight.items()]

        logging.debug(f"loss_weights: {loss_weights}")

        (balanced_train_datagen, balanced_val_datagen, balanced_test_datagen) = get_balanced_data_generators(
            images_dir, batch_size=batch_size, preprocess_fn=tf.keras.applications.resnet50.preprocess_input)

        callback_lr = LossLearningRateScheduler(base_lr=0.001, lookback_epochs=3)
        best_val_loss_model_checkpoint = ModelCheckpoint(best_val_loss_save_path, monitor="val_loss",
                                                         mode="min", save_best_only=True, verbose=1)
        opt = Adam(learning_rate=learning_rate, decay=learning_rate / epochs)
        self.model.compile(loss=[categorical_focal_loss(alpha=[loss_weights])], optimizer=opt, metrics=["accuracy"])
        self.model.fit(balanced_train_datagen, validation_data=balanced_val_datagen, epochs=epochs,
                  # callbacks=[best_val_loss_model_checkpoint, EarlyStopping(monitor="val_loss", mode="min", baseline="0.01", patience=20)],
                  callbacks=[best_val_loss_model_checkpoint, callback_lr],
                  verbose=2)
        self.model.evaluate(balanced_test_datagen)
        tf.keras.models.save_model(self.model, model_save_path)


    def get_identity_block(self, x, filter, kernel=(3, 3)):
        raise Exception("Not implemented!")


    def get_conv_block(self, x, filter, kernel=(3, 3), strides=(2, 2)):
        raise Exception("Not implemented")


    def get_data_generator(self, df_data, subset="training", val_split=0.2, batch_size=32):
        data_gen = ImageDataGenerator(
            rotation_range=15,
            brightness_range=(0.9, 1.1),
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=val_split,
            preprocessing_function=tf.keras.applications.resnet50.preprocess_input
        )
        data_gen_iter = data_gen.flow_from_dataframe(df_data,
                                                     directory=None,
                                                     x_col="imagepath",
                                                     y_col="category",
                                                     weight_col="weight",
                                                     color_mode="rgb",
                                                     class_mode="categorical",
                                                     batch_size=batch_size,
                                                     save_to_dir=None,
                                                     subset=subset,
                                                     validate_filenames=False)

        return data_gen_iter


    def preprocess_image(self, img):
        logging.debug("Resnet image preprocessing ...")
        return tf.keras.applications.imagenet_utils.preprocess_input(img, mode="caffe")


    def get_predicted_probability(self,  prediction_result):
        logging.debug(f"Resnet probability: {prediction_result}")
        return prediction_result[0][1]



