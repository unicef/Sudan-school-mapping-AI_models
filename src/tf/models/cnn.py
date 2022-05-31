from src.tf.models.common import BaseModel


class CNNBinaryClassifierForLocalization(BaseModel):

    def __init__(self,
                 input_shape=(92, 92, 3),
                 stride_x=15,
                 stride_y=15,
                 max_boxes_num=3,
                 *args,
                 **kwargs):
        """

        :param input_shape: The shape of images that are used for the model training
        :param stride_x: the amount of movement of sliding window over an image along x-axis
        :param stride_y: the amount of movement of sliding window over an image along y-axis
        :param max_boxes_num: how many bounding boxes to keep when calculating the
        boundaries of the container bounding box
        """
        super().__init__(stride_x=stride_x,
                         stride_y=stride_y,
                         input_shape=input_shape,
                         schools_threshold=0.95,
                         *args, **kwargs)
        self.max_boxes_num = max_boxes_num

    def train(self):
        pass

    def predict(self, img):
        """
        Runs prediction on an image.
        :param img: PIL image object
        :return: a coordinates for the bounding box containing the child bounding boxes with the highest
        probability that they contain school buildings.
        """
        results = super().predict(img)
        boxes_with_schools = list(filter(lambda r: r["is_school"], results))
        if len(boxes_with_schools) > 0:
            boxes_to_keep = self.get_top_probability_boxes(boxes_with_schools, self.max_boxes_num)
            container_boundaries = self.get_container_boundaries(boxes_to_keep)
            return container_boundaries
        return None
