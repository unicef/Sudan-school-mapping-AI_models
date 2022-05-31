import os
import logging
from pathlib import Path
from typing import Tuple, List
import src.constants as constants
import src.tf.models.loss_utils as loss_utils
from tensorflow.keras.preprocessing import image
import src.tf.models.models_utils as models_utils
from src.bbox_utils import filter_overlapping_predictions
from src.tf.models.object_localizer import ObjectLocalizer
from src.tf.models.object_classifier import ObjectClassifier


class EnsembleModel():

    def __init__(self,
                 classifier_model_path: str=None,
                 localizer_model_path: str=None,
                 output: str=constants.BBOXES_OUTPUT_DIR,
                 tile_shape: Tuple=(256, 256),
                 skip_localization: bool=False,
                 **kwargs):
        """
        :param binary_classifier_model_path: the path to the saved model or weights file for the binary classifier model.
        :param object_localizer_model_path: the path to the saved model or weights file for the object localization model.
        :param output: the path to the folder where prediction files with be saved
        :param tile_shape: a shape of the image tile
        :param skip_localization: True if just object classification is run othwerwise False
        """
        classifier_ctr = models_utils.get_model_constructor(classifier_model_path)
        localizer_ctr = models_utils.get_model_constructor(localizer_model_path)
        self.binary_classifier = classifier_ctr(model_path=classifier_model_path, **kwargs)
        self.object_localizer = localizer_ctr(model_path=localizer_model_path, **kwargs)
        self.output_dir = output
        self.tile_shape = tile_shape
        self.skip_localization = skip_localization


    def predict(self, img_path: str) -> None:
        """
        Runs prediction on an image and saves the bounding box boundaries into a file.

        :param img_path: a file path to an image
        """
        if not os.path.exists(img_path):
            raise f"Image at path {img_path} does not exist."
        img = image.load_img(img_path)
        original_width, original_height = img.size
        results = self.binary_classifier.predict(img)
        tiles_with_schools = list(filter(lambda r: r["is_school"], results))
        logging.debug(f"Binary classifier results: {tiles_with_schools}")
        predictions = []
        # If there are tiles with schools run bounding boxes prediction on them.
        if not self.skip_localization and len(tiles_with_schools) > 0:
            img_arr = image.img_to_array(img)
            for tile in tiles_with_schools:
                img_slice = img_arr[tile["top"]:tile["bottom"], tile["left"]:tile["right"]]
                predicted_bboxes = self.object_localizer.predict(img_slice)
                logging.debug(predicted_bboxes)
                if predicted_bboxes:
                    for bbox_pred in predicted_bboxes:
                        # Calculate the absolute coordinates of bounding boxes compared to an image coordinate space.
                        # Calculated values must not be greater than the original images dimensions.
                        bbox_info = [
                            tile["probability"],
                            bbox_pred["probability"],
                            tile["left"]+bbox_pred["left"],
                            tile["top"]+bbox_pred["top"],
                            tile["left"]+bbox_pred["right"],
                            tile["top"]+bbox_pred["bottom"]
                        ]
                        predictions.append(bbox_info)
                else:
                    predictions.append([tile["probability"],tile["left"],tile["top"],tile["right"],tile["bottom"]])
        filtered_predictions = filter_overlapping_predictions(predictions)
        logging.debug(f"kept predictions: {filtered_predictions}")
        img_name = os.path.basename(img_path)
        self.save_results(filtered_predictions, img_name)


    def save_results(self, predictions: str, img_name: str) -> None:
        """
        Saves binary classification and/or object localization probability and bounding boxes information for
        specified image into a .txt file.

        :param predictions a list of probabilities and bounding boxes coordinates for the specified image.
        :param img_name: an image on which the prediction is run
        """
        if predictions:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            out_file = f"{os.path.splitext(img_name)[0]}.txt"
            out_path = os.path.join(self.output_dir, out_file)
            logging.debug(f"Writting output to {out_path}")
            with open(out_path, "w") as fp:
                for p in predictions:
                    if len(p) > 5:
                        fp.write(f"{p[0]},{p[1]},{p[2]},{p[3]},{p[4]},{p[5]}\n")
                    else:
                        fp.write(f"{p[0]},{p[1]},{p[2]},{p[3]},{p[4]}\n")



