import os
import sys
import logging
from src.tf.models.ensemble import EnsembleModel


def main(args):
    model = EnsembleModel(**vars(args))
    images = os.listdir(args.input)
    images_len = len(images)
    for index, img_name in enumerate(images):
        img_path = os.path.join(args.input, img_name)
        try:
            logging.info(f"Running prediction for {img_name} ...")
            model.predict(img_path)
            logging.info(f"{images_len-index-1} image(s) left.")
        except Exception as ex:
            logging.error(ex)
            raise(ex)
            continue