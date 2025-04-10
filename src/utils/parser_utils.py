from argparse import Namespace

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n', 'off'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y', 'on'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def update_from_config(args: Namespace, config: dict):
    # assert set(config.keys()) <= set(vars(args)), f'{set(config.keys()).difference(vars(args))} not in args.'
    args.__dict__.update(config)
    return args

def config_dict_from_args(args):
    """
    Extract a dictionary with the experiment configuration from arguments (necessary to filter TestTube arguments)

    :param args: TTNamespace
    :return: hyparams dict
    """
    keys_to_remove = {'hpc_exp_number', 'trials', 'optimize_parallel', 'optimize_parallel_gpu',
                      'optimize_parallel_cpu', 'generate_trials', 'optimize_trials_parallel_gpu'}
    hparams = {key: v for key, v in args.__dict__.items() if key not in keys_to_remove}
    return hparams