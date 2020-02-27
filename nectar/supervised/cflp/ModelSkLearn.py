import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from nectar.utils import save_pickle
from .ModelCFLP import ModelCFLP
import time


class ModelSkLearn(ModelCFLP):
    def __init__(self, model_config, path):
        super(ModelSkLearn, self).__init__(model_config, path)
        self.model = self._sklearn_model_factory()
        self.x_train, self.x_test = self._prepare_x(ModelCFLP.x_train_raw), \
                                    self._prepare_x(ModelCFLP.x_test_raw)
        self.y_train, self.y_test = self._prepare_y(ModelCFLP.y_train_raw), \
                                    self._prepare_y(ModelCFLP.y_test_raw)
        self.y_train_pred_raw, self.y_test_pred_raw = None, None

    def train(self):
        print(f"Started training {self.config['name']} model")
        self.model.fit(self.x_train, self.y_train)
        # save model
        model_path = self.exp_path / "model.pkl"
        save_pickle(model_path, self.model)

    def predict(self):
        print("Predicting...")
        self.y_train_pred = self.model.predict(self.x_train)
        self.y_train_pred[self.y_train_pred < 0] = 0
        self.y_train_pred_raw = {item["pid"]: self.y_train_pred[idx] for idx, item in enumerate(ModelCFLP.x_train_raw)}

        start_time = time.time()
        self.y_test_pred = self.model.predict(self.x_test)
        self.time_prediction = (time.time() - start_time) / self.x_test.shape[0]
        self.y_test_pred[self.y_test_pred < 0] = 0
        self.y_test_pred_raw = {item["pid"]: self.y_test_pred[idx] for idx, item in enumerate(ModelCFLP.x_test_raw)}

    def evaluate_learning_metrics(self):
        print("Evaluating learning metrics...")

        self.metrics = self._calculate_metrics()
        print(f'Train mse: {self.metrics["train_loss"]}, '
              f'Test mse: {self.metrics["test_loss"]} '
              f'Train r2: {self.metrics["train_r2"]}, '
              f'Test r2: {self.metrics["test_r2"]}')
        # self._plot_demand_comparison()
        # self._plot_xi_histogram()

    def save(self):
        super().save()
        save_pickle(self.exp_path / "y_test_pred_raw.pkl", self.y_test_pred_raw)
        save_pickle(self.exp_path / "y_train_pred_raw.pkl", self.y_train_pred_raw)
        save_pickle(self.exp_path / "time_prediction.pkl", self.time_prediction)
        # Save model
        # model_path = self.exp_path / "model.pkl"
        # save_pickle(model_path, self.model)

    def load(self):
        pass

    @staticmethod
    def _extract_data(method, x):
        _x = np.hstack((x['c_f_normalized'].reshape(-1),
                        x['c_v_normalized'].reshape(-1)))
        if method == "scen" or method == "both":
            _x = np.hstack((_x, x['scenario_normalized'].reshape(-1)))
        if method == "feat" or method == "both":
            _x = np.hstack((_x, x['scenario_features'].reshape(-1)))

        return _x

    def _prepare_x(self, x_input):
        x = []
        for i in x_input:
            if self.config["scen"] and not self.config["feat"]:
                x.append(self._extract_data("scen", i))
            elif not self.config["scen"] and self.config["feat"]:
                x.append(self._extract_data("feat", i))
            elif self.config["scen"] and self.config["feat"]:
                x.append(self._extract_data("both", i))
            else:
                print("Need atleast scenario or features in the input")

        return np.asarray(x)

    @staticmethod
    def _prepare_y(y_input):
        y = [i["xi_hat"] for i in y_input]
        return np.asarray(y)

    def _sklearn_model_factory(self):
        if self.config["name"] == "linear":
            return LinearRegression()
        elif self.config["name"] == "ridge":
            return Ridge(alpha=self.config["alpha"])
        elif self.config["name"] == "lasso":
            return Lasso(alpha=self.config["alpha"])
        elif self.config["name"] == "elastic":
            return ElasticNet()
        elif self.config["name"] == "decision_tree":
            return DecisionTreeRegressor(
                max_depth=self.config['max_depth'],
                max_features=self.config['max_features'])

    def _calculate_metrics(self):
        return {"train_loss": mean_squared_error(self.y_train, self.y_train_pred),
                "test_loss": mean_squared_error(self.y_test, self.y_test_pred),
                "train_r2": r2_score(self.y_train, self.y_train_pred),
                "test_r2": r2_score(self.y_test, self.y_test_pred)}
