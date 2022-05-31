import numpy as np
from typing import Tuple, List
from src.tf.models.common import BaseModel


class ObjectLocalizer(BaseModel):

    def __init__(self,
                 input_shape: Tuple[int, int, int]=(92, 92, 3),
                 stride_x: int=46,
                 stride_y: int=46,
                 *args,
                 **kwargs):
        """
        :param input_shape: The shape of images that are used for the model training
        :param stride_x: the sliding window step size in pixels along the x-axis
        :param stride_y: the sliding window step size in pixels along the y-axis
        """
        super().__init__(stride_x=stride_x,
                         stride_y=stride_y,
                         input_shape=input_shape,
                         *args, **kwargs)


    def train(self):
        pass


    def build(self):
        pass


    def get_predicted_probability(self, prediction_result: List[List]) -> float:
        return prediction_result[0][0]


    def predict(self, img: np.array) -> List[dict]:
        """
        Runs prediction on an image.
        :param img: PIL image object
        :return: a list of dictionaries containing information about each bounding box that contains school.
        """
        results = super().predict(img)
        boxes_with_schools = list(filter(lambda r: r["is_school"], results))
        return boxes_with_schools