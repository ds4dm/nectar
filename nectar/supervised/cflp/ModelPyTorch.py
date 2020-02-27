import numpy as np
import torch as T
import torch.nn as nn
from comet_ml import Experiment
from sklearn.metrics import r2_score
import time
from nectar.supervised.cflp.utils import EarlyStopping
from nectar.supervised.cflp.utils import create_models_and_optimizers
from nectar.supervised.cflp.utils import get_data_loaders
from nectar.supervised.cflp.utils import prepare_data
from nectar.supervised.cflp.utils import save_models
from nectar.supervised.cflp.utils import step
from nectar.supervised.cflp.utils import zero_grad
from nectar.utils import save_pickle
from .ModelCFLP import ModelCFLP

T.manual_seed(0)


class ModelPyTorch(ModelCFLP):
    def __init__(self, config, path):
        super(ModelPyTorch, self).__init__(config, path)
        model_args_dict = {k: v for k, v in self.config.items() if 'phi' not in k and 'rho' not in k}
        self.config = self.config if self.config["use_deepset"] else model_args_dict

        self.device = T.device("cuda:2" if T.cuda.is_available() else "cpu")
        self.models, self.optimizers = create_models_and_optimizers(self.config, self.device)
        self._set_criterion()
        self._set_data_loaders()
        self._set_early_stopping()
        self._set_comet_exp()

        self.train_losses, self.train_r2 = [], []
        self.val_losses, self.val_r2 = [], []

    def train(self):
        if self.config["early_stop"]:
            self.early_stopping.exp_id = self.exp_path

        for epoch in range(1, self.config["num_epochs"] + 1):
            # Train model
            train_epoch_loss, train_epoch_r2 = self._train_epoch(epoch)
            self.train_losses.append(train_epoch_loss), self.train_r2.append(train_epoch_r2)

            # Validate model
            val_epoch_loss, val_epoch_r2, _, _ = self._val_epoch(epoch)
            self.val_losses.append(val_epoch_loss), self.val_r2.append(val_epoch_r2)

            if self.config["use_comet"]:
                with self.comet_exp.train():
                    self.comet_exp.log_metric('loss_epoch', train_epoch_loss, step=epoch)
                with self.comet_exp.validate():
                    self.comet_exp.log_metric('loss_epoch', val_epoch_loss, step=epoch)

            print(f"Epoch {epoch}: Train loss {train_epoch_loss:.3f}, Val loss {val_epoch_loss:.3f} "
                  f"Train R2 {train_epoch_r2:.3f}, Val R2 {val_epoch_r2:.3f}")

            if self.config["early_stop"]:
                should_stop, early_stop_epoch = self.early_stopping.step(epoch,
                                                                         val_epoch_loss,
                                                                         self.models)
                if should_stop:
                    print("Early stopping ...")
                    break

        min_val_idx = int(np.argmin(np.asarray(self.val_losses)))
        self.metrics["train_loss"] = self.train_losses[min_val_idx]
        self.metrics["val_loss"] = self.val_losses[min_val_idx]
        self.metrics["train_r2"] = self.train_r2[min_val_idx]
        self.metrics["val_r2"] = self.val_r2[min_val_idx]

        print("Finish training")
        # save_models(args, exp_id, models)
        print(f"** Min validation loss: {self.metrics['val_loss']} at epoch {min_val_idx + 1}")

    def predict(self):
        self.test_loss, self.test_r2, self.y_test_pred, self.time_prediction = self._val_epoch(0, split="test")
        self.y_test_pred[self.y_test_pred < 0] = 0
        self.y_test_pred_raw = {item["pid"]: self.y_test_pred[idx] for idx, item in enumerate(ModelCFLP.x_test_raw)}

    def evaluate_learning_metrics(self):
        self.metrics["test_loss"] = self.test_loss
        self.metrics["test_r2"] = self.test_r2
        # self._plot_xi_histogram()

    def save(self):
        super().save()
        # Log train/val loss
        save_pickle(self.exp_path / "y_test_pred_raw.pkl", self.y_test_pred_raw)
        save_pickle(self.exp_path / "time_prediction.pkl", self.time_prediction)
        np.save(self.exp_path / 'train_loss.npy', np.asarray(self.train_losses))
        np.save(self.exp_path / 'val_loss.npy', np.asarray(self.val_losses))

        save_models(self.config, self.exp_path, self.models)

    def load(self):
        pass

    def _set_criterion(self):
        if self.config["criterion"] == "mse":
            self.criterion = nn.MSELoss()
        elif self.config["criterion"] == "sl1":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError("Invalid criterion")

    def _set_comet_exp(self):
        if self.config['use_comet']:
            self.comet_exp = Experiment(api_key="2G5zZt4hxraqN2t4zysCo7tYu",
                                        project_name="supervised-xi",
                                        workspace="rahulptel",
                                        auto_metric_logging=False)
            # Create update args and create comet experiment
            self.config.update({"model_cfg": "[" + ", ".join(list(map(str, self.config["model_cfg"]))) + "]"})
            self.comet_exp.log_parameters(self.config)

    def _set_data_loaders(self):
        self.train_loader, self.val_loader, self.test_loader = \
            get_data_loaders(self.config, ModelCFLP.path["data"])

    def _set_early_stopping(self):
        if self.config["early_stop"]:
            self.early_stopping = EarlyStopping(self.config)

    def _train_epoch(self, epoch):
        train_losses = 0
        train_r2 = 0
        train_samples = 0

        if self.config["use_scenario"] and self.config["use_deepset"]:
            model, phi, rho = self.models
            opt_model, opt_phi, opt_rho = self.optimizers
            phi.train(), rho.train()
        else:
            model = self.models
            opt_model = self.optimizers
        model.train()

        for batch_idx, batch in enumerate(self.train_loader):
            opt_model.zero_grad()

            x, y = prepare_data(batch, self.config, self.models, self.device)
            zero_grad(self.config, self.optimizers)
            y_pred = model(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            step(self.config, self.optimizers)

            loss_item = loss.item()
            train_losses += loss_item * x.shape[0]
            train_r2 += r2_score(y.cpu().numpy(), y_pred.clone().detach().cpu().numpy()) * x.shape[0]
            train_samples += x.shape[0]

            # if (batch_idx + 1) % 10 == 0:
            #     print(f"Epoch {epoch}: [{batch_idx * x.shape[0]}:{len(loader.data_manager)}] "
            #           f"Train loss {loss.item():.3f}")

        return train_losses / train_samples, train_r2 / train_samples

    def _val_epoch(self, epoch, split="val"):
        losses = 0
        r2 = 0
        samples = 0
        xi_hat_pred = []

        if self.config["use_scenario"] and self.config["use_deepset"]:
            model, phi, rho = self.models
            phi.eval(), rho.eval()
        else:
            model = self.models
        model.eval()

        loader = self.val_loader if split == "val" else self.test_loader
        start_time = time.time()
        with T.no_grad():
            for batch_idx, batch in enumerate(loader):
                x, y = prepare_data(batch, self.config, self.models, self.device)

                y_pred = model(x)
                loss = self.criterion(y_pred, y)

                loss_item = loss.item()
                losses += loss_item * x.shape[0]
                r2 += r2_score(y.cpu().numpy(), y_pred.clone().detach().cpu().numpy()) * x.shape[0]
                samples += x.shape[0]

                xi_hat_pred.append(y_pred.cpu().numpy())
                # if batch_idx + 1 % 10 == 0:
                #     print(f"Epoch {epoch}: [{batch_idx * x.shape[0]}:{len(loader.data_manager)}] "
                #           f"{eval_type} loss {loss.item():.3f}")
        total_time = time.time() - start_time
        xi_hat_pred = np.concatenate(xi_hat_pred)

        return losses / samples, r2 / samples, xi_hat_pred, total_time / len(loader)
