import logging
import importlib
import src.cli as cli
import src.constants as constants
from src.utils import get_command_args


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = cli.ARGS
    if args.model_name not in constants.PARAMS["models"]:
        raise ValueError(f"Model name {args.model_name} is not supported.")
    if hasattr(args, "stage_name") and \
        args.stage_name and \
        args.stage_name not in constants.PARAMS["stages"].keys():
        raise ValueError(f"The stage {args.stage_name} does not exists in the allowed list of pipeline stages.")
    stages = constants.PARAMS["stages"]
    if hasattr(args, "stage_name") and args.stage_name:
        stages = {args.stage_name:stages[args.stage_name]}
    for key, value in stages.items():
        stage_data = constants.PARAMS["stages"][key]
        command_args = get_command_args(dict.setdefault(stage_data, "params"), cli.ARGS)
        module_obj = importlib.import_module(stage_data["run"])
        getattr(module_obj, "main")(command_args)

