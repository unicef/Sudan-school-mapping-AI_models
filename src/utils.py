from argparse import Namespace


def get_command_args(default_args: dict, actual_args: Namespace):
    """
    Merges default and actual arguments where new arguments will take precedence over the default ones.
    :param default_args: default CLI arguments
    :param actual_args: actual CLI arguments
    :return: merged arguments
    """
    args = Namespace()
    if default_args:
        for key, val in default_args.items():
            val = getattr(actual_args, key) \
                if hasattr(actual_args, key) and getattr(actual_args, key) \
                else default_args[key]
            setattr(args, key, val)
    # If there are command line arguments that are not processed add them to
    # the arguments dictionary.
    args_diffs = set(vars(actual_args))-set(vars(args))
    for a in args_diffs:
        setattr(args, a, getattr(actual_args, a))
    return args


