import hashlib
import pickle as pkl
from abc import ABC, abstractmethod


def create_config_hash(config):
    """Create a hash based on configuration values

    Creating a unique hash based on the values of the configuration
    helps us define a scope around the data. Hence

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration for the pipeline

    Returns
    -------
    value_hash : str
        Hash representing the current pipeline configuration
    """
    value_str = ""
    for section in config.sections:
        for key in section.keys():
            value_str += str(config[section][key])
    value_hash = hashlib.md5(value_str.encode('utf-8')).hexdigest()

    return value_hash


def load_pickle(path, check=True):
    if check:
        assert path.exists(), print(f"{str(path)} not found!")

    with open(path, 'rb') as fp:
        return pkl.load(fp)


def save_pickle(path, data, check=True):
    if check:
        assert path.parent.exists(), print(f"{str(path.parent)} not found!")

    with open(path, "wb") as fp:
        pkl.dump(data, fp)


class SIPP(ABC):
    """
    Class for Stochastic Integer Programming Problem
    """

    @abstractmethod
    def set_xi_bar(self, xi_bar):
        pass

    @abstractmethod
    def make_two_sip_model(self, use_xi_bar=False):
        pass

    @abstractmethod
    def make_second_stage_model(self, sol, xi):
        pass

    @abstractmethod
    def solve_two_sip(self, gap=0.02, time_limit=600, threads=2):
        pass

    @abstractmethod
    def get_second_stage_objective(self, sol, xi, threads=2):
        pass

    @abstractmethod
    def evaluate_x(self, sol):
        pass
