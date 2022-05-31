import os
import math
import numpy as np
import tensorflow as tf
from typing import List
from pathlib import Path
from datetime import datetime
import src.constants as constants
from abc import ABC, abstractmethod
from tensorflow.keras.preprocessing import image


class BaseModel(ABC):

    def __init__(self,
                 input_shape=(256, 256, 3),
                 model_path=None,
                 schools_threshold=0.6,
                 stride_x=128,
                 stride_y=128,
                 debug=True,
                 **kwargs):
        """
        :param input_shape: The shape of images that are used for the model training
        :param model_path: The path to the file containing the saved model or just model weights
        :param schools_threshold: Schools threshold probability used during prediction.
        :param stride_x: the amount of movement of sliding window over an image along x-axis
        :param stride_y: the amount of movement of sliding window over an image along y-axis
        :param debug: if true then save image slices on which predictions are run on
        """
        if model_path:
            if model_path.endswith(".h5"):
                self.model = self.build()
                if self.model is not None:
                    self.model.load_weights(model_path)
                else:
                    self.model = tf.keras.models.load_model(model_path,
                                                            custom_objects=kwargs.setdefault("custom_objects", None))
            else:
                self.model = tf.keras.models.load_model(model_path,
                                                        custom_objects=kwargs.setdefault("custom_objects", None))
        self.input_shape = input_shape
        self.schools_threshold = schools_threshold
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.debug = debug


    @abstractmethod
    def build(self):
        pass


    @abstractmethod
    def train(self):
        pass


    @abstractmethod
    def get_predicted_probability(self, prediction_result: List[List]):
        pass


    def preprocess_image(self, img):
        return img/255.0


    def get_model_save_path(self, prefix="model"):
        model_name = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
        return model_name


    def predict(self, img):
        """
        Splits an image into tiles if necessary (size > 256) and runs prediction on each tile.
        :param img: PIL image
        :return: an array of prediction results for each image tile.
        """
        img_arr = image.img_to_array(img)
        tiles_y = math.ceil((img_arr.shape[0] - self.input_shape[0]) / self.stride_y) + 1
        tiles_x = math.ceil((img_arr.shape[1] - self.input_shape[1]) / self.stride_x) + 1
        max_pos_y = img_arr.shape[0] - self.input_shape[0]
        max_pos_x = img_arr.shape[1] - self.input_shape[1]
        results = []

        for ty in range(tiles_y):
            for tx in range(tiles_x):
                start_y = min(max_pos_y, ty * self.stride_y)
                start_x = min(max_pos_x, tx * self.stride_x)
                end_y = min(img_arr.shape[0], start_y + self.input_shape[0])
                end_x = min(img_arr.shape[1], start_x + self.input_shape[1])
                img_slice = img_arr[start_y:end_y, start_x:end_x]
                if self.debug:
                    Path(constants.TEMP_IMAGES_PATH).mkdir(parents=True, exist_ok=True)
                    hash_str = abs(hash(str(img_slice[:10])))
                    image.save_img(os.path.join(constants.TEMP_IMAGES_PATH, f"{ty}_{tx}_{hash_str}.png"), img_slice)
                img_slice = self.preprocess_image(img_slice)
                images_batch = np.expand_dims(img_slice, axis=0)
                classes_prob = self.model.predict(images_batch)
                prob = self.get_predicted_probability(classes_prob)
                has_school = prob > self.schools_threshold
                pred_data = {"is_school": has_school,
                             "probability": prob,
                             "left": start_x,
                             "top": start_y,
                             "right": end_x,
                             "bottom": end_y}
                #if has_school:
                #    print(f"Found tile with school: {pred_data}")
                results.append(pred_data)

        return results