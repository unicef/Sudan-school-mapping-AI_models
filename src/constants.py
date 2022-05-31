import os
import yaml


def _get_params(yaml_file_path):
    with open(yaml_file_path) as fp:
        params = yaml.safe_load(fp)
    return params


# Path to a folder containing image patches created during debugging.
TEMP_IMAGES_PATH=os.path.join(os.path.dirname(__file__), "..", "data", "temp")

# Path to a folder with files containing bounding box boundaries predictions.
BBOXES_OUTPUT_DIR=os.path.join(os.path.dirname(__file__), "..", "data", "bbox_predictions")

# Parameters from the params.yaml file.
PARAMS = _get_params(os.path.join(os.path.dirname(__file__), "..", "params.yaml"))