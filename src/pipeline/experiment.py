import os
import warnings
import numpy as np
import  argparse

from src.utils import parser_utils

class Experiment:
    r"""
    Simple class to handle the routines used to run experiments.

    Args:
        run_fn: Python function that actually runs the experiment when called.
                The run function must accept single argument being the experiment hyperparameters.
        parser: Parser used to read the hyperparameters for the experiment.
        config_path: Path to configuration files, if not specified the default will be used.
    """
    def __init__(self, run_fn, parser: argparse, config_path=None):
        self.run_fn = run_fn
        self.parser = parser
        self.config_root = config_path

    def _check_config(self, hparams):
        config_file = hparams.__dict__.get('config', None)
        if config_file is not None:
            # read config file
            import yaml

            config_file = os.path.join(self.config_root, config_file)
            with open(config_file, 'r') as fp:
                experiment_config = yaml.load(fp, Loader=yaml.FullLoader)

            # update hparams
            hparams = parser_utils.update_from_config(hparams, experiment_config)
            if hasattr(self.parser, 'parsed_args'):
                self.parser.parsed_args.update(experiment_config)
        return hparams

    def make_run_dir(self):
        """Create directory to store run logs and artifacts."""
        raise NotImplementedError

    def run(self):
        hparams = self.parser.parse_args()
        hparams = self._check_config(hparams)

        return self.run_fn(hparams)

    def run_many_times_sequential(self, n):
        hparams = self.parser.parse_args()
        hparams = self._check_config(hparams)
        warnings.warn('Running multiple times. Make sure that randomness is handled properly')
        for i in range(n):
            print(f"**************Trial n.{i}**************")
            np.random.seed()
            self.run_fn(hparams)

    def deal_data(self):
        hparams = self.parser.parse_args()
        hparams = self._check_config(hparams)
        return self.run_fn(hparams)