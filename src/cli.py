import yaml
import argparse
import src.constants as constants
from collections import namedtuple


def parse_args(params):
    """
    Parses command line options and returns parsed options values.
    :param params: default values from the params.yaml file
    :return: a namespace with parsed options and values
    """
    parser = argparse.ArgumentParser(description="Process pipeline arguments.")
    parser.add_argument("-m", dest="model_name", default="yolo", help="the model name: tf or yolo")
    parser.add_argument("-i", dest="input", default=None, help="the path to a folder or a file containing required input for a pipeline stage or from a model.")
    parser.add_argument("-o", dest="output", default=None, help="the path to the folder with files containing output from a pipeline stage or from a model.")
    parser.add_argument("-n", dest="stage_name", default=None, help="the name of the pipeline stage.")
    args = parser.parse_args()
    return args


# Contains parsed command line options and values
ARGS = parse_args(constants.PARAMS)