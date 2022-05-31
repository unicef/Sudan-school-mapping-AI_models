import logging
import importlib
import src.constants as constants
from argparse import Namespace
from src.utils import get_command_args


def main(args: Namespace):
    model_data = constants.PARAMS["models"][args.model_name]
    command_to_run_model = model_data["run"]
    command_args = get_command_args(dict.setdefault(model_data, "params"), args)
    module_obj = importlib.import_module(command_to_run_model)
    logging.debug(f"command_args: {command_args}")
    getattr(module_obj, "main")(command_args)