import hashlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch as T
from PIL import Image

from ..utils import save_pickle


class Model():
    # Static. Shared across all instances
    path = None
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = None, None, None, None
    test_pids = None

    def __init__(self, config, path):
        self.config = config
        self.y_train_pred, self.y_test_pred = np.asarray([]), np.asarray([])
        self.metrics = {}
        self.exp_path = None

        Model.path = path
        Model.x_train_raw is None and self._set_static_data()

    @staticmethod
    def _set_static_data():
        """Set the following static class variables:

        Model.x_train : npy
            Train input data
        Model.x_test : npy
            Test input data
        Model.y_train : npy
            Train output data
        Model.y_test : npy
            Test output data
        """
        # Training data
        Model.x_train_raw = np.load(Model.path["data"] / "x_train_raw.npy", allow_pickle=True)
        Model.x_test_raw = np.load(Model.path["data"] / "x_test_raw.npy", allow_pickle=True)
        Model.y_train_raw = np.load(Model.path["data"] / "y_train_raw.npy", allow_pickle=True)
        Model.y_test_raw = np.load(Model.path["data"] / "y_test_raw.npy", allow_pickle=True)
        # Model.xi_average = np.load(Model.path["data"] / "x_test_xi_avg.npy")

    def create_experiment_folder(self):
        # Create experiment folder
        time_stamp = datetime.now().strftime("%d-%b-%Y_%H:%M:%S.%f")
        exp_id = ""
        for k, v in self.config.items():
            exp_id += k + str(v)
        exp_id = hashlib.md5(exp_id.encode('utf-8')).hexdigest()
        exp_id = time_stamp + "_" + exp_id

        self.exp_path = Model.path["data"] / "experiments" / exp_id
        if not self.exp_path.exists():
            self.exp_path.mkdir(parents=True, exist_ok=True)

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate_learning_metrics(self):
        pass

    def save(self):
        save_pickle(self.exp_path / 'config.pkl', self.config)
        save_pickle(self.exp_path / "metrics.pkl", self.metrics)

    def load(self):
        pass

    @staticmethod
    def _convert_fig_to_tensor(canvas):
        # Option 2: Save the figure to a string.
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()

        # Option 2b: Pass off to PIL.
        im = Image.frombytes("RGBA", (width, height), s)
        im_np = np.asarray(im.convert('RGB')).transpose((2, 0, 1))

        im_tensor = T.from_numpy(im_np)
        return im_tensor

    def _plot_xi_histogram(self):
        fig = plt.figure()
        print(Model.y_test.shape, self.y_test_pred.shape, Model.xi_average.shape)
        plt.hist(Model.y_test.reshape(-1), bins=80, alpha=0.5, label="orig")
        plt.hist(Model.xi_average.reshape(-1), bins=80, alpha=0.5, label="avg")
        plt.hist(self.y_test_pred.reshape(-1), bins=80, alpha=0.5, label="pred")
        plt.legend()
        plt.savefig(self.exp_path.joinpath('xi_hist.jpg'))
        plt.close(fig)
