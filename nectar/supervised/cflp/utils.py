"""
Module containing utility functions - for the regression task of predicting xi_hat -
of train module.
"""
import ast
import copy
import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

np.random.seed(0)
T.manual_seed(0)


class NectarDataset(Dataset):
    """
    A Pytorch Dataset for the regression task to predict xi_hat

    Attributes
    ----------
    x : Numpy array containing dicts
        Feature set for the regression task
    y : Numpy 2D array
        Labels for the the regression task

    Methods
    -------
    __len__()
        Returns the number of samples in the data_manager
    __getitem__(i)
        Returns the training item {x, y} at ith index
    """

    def __init__(self, data_path, split="train"):
        self.x = np.load(data_path / "".join(["x_", split, "_raw.npy"]), allow_pickle=True)
        self.y = np.load(data_path / "".join(["y_", split, "_raw.npy"]), allow_pickle=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"pid": self.x[i]["pid"],
                "c_f": self.x[i]["c_f_normalized"],
                "c_v": self.x[i]["c_f_normalized"],
                "scenario": self.x[i]["scenario_normalized"],
                'features': self.x[i]["scenario_features"],
                'y': self.y[i]["xi_hat"]}


def get_data_loaders(config, data_path):
    # Create data_manager
    train_dataset = NectarDataset(data_path, split="train")
    test_dataset = NectarDataset(data_path, split="test")
    print("Train data_manager :", len(train_dataset), " Test data_manager :", len(test_dataset))

    idxs = [i for i in range(len(train_dataset))]
    np.random.shuffle(idxs)
    train_sampler = SubsetRandomSampler(idxs[:15])
    val_sampler = SubsetRandomSampler(idxs[15:])

    # Create loaders
    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=config['bs'])
    val_loader = DataLoader(train_dataset,
                            sampler=val_sampler,
                            batch_size=config['bs'])
    test_loader = DataLoader(test_dataset,
                             batch_size=config['bs'],
                             shuffle=False)
    print("Train loader :\t", len(train_loader))
    print("Val loader :\t", len(val_loader))
    print("Test loader :\t", len(test_loader))

    return train_loader, val_loader, test_loader


class FeedForwardNet(nn.Module):
    """
    A simple Feed Forward Neural Network with specified input, hidden and
    output dimensions and hidden activation.

    As per the current implementation
        - No Dropout in the input and output layer
        - No Batch Normalization in the output layer
    """

    def __init__(self,
                 cfg,
                 dp=0,
                 activation="relu",
                 weight_init="xav_uni",
                 use_dropout=False,
                 use_batch_norm=True):
        super(FeedForwardNet, self).__init__()
        self.activation = activation
        self.weight_init = weight_init
        self.use_batch_norm = use_batch_norm
        self.num_layers = len(cfg) - 1
        self.dropout = nn.Dropout(p=dp)

        self.layers = nn.ModuleList()
        if use_batch_norm:
            self.batch_norm = nn.ModuleList()
        self.batch_norm.append(nn.BatchNorm1d(cfg[0]))

        for i in range(self.num_layers):
            layer = nn.Linear(cfg[i], cfg[i + 1])
            init_layer_weights(layer, self.weight_init, self.activation)
            self.layers.append(layer)
            if i != self.num_layers - 1 and use_batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(cfg[i + 1]))

    def forward(self, x):
        x = self.batch_norm[0](x)
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)  # Linear
            x = self.batch_norm[i + 1](x) if self.use_batch_norm else x  # BN
            x = self.dropout(x) if i > 0 else x  # Dropout
            x = {
                "relu": F.relu(x),
                "leaky_relu": F.leaky_relu(x),
                "elu": F.elu(x),
                "tanh": T.tanh(x),
                "sigmoid": T.sigmoid(x)
            }.get(self.activation, None)  # Activation
            assert x is not None, f"Unknown activation {self.activation}"

        # Last layer
        x = self.layers[-1](x)  # Linear
        x = F.relu(x)

        return x


class SkipBlock(nn.Module):
    """
    Block with a linear layer and skip connection
    """

    def __init__(self, size):
        super(SkipBlock, self).__init__()
        self.size = size

        self.linear = nn.Linear(size, size)
        init_layer_weights(self.linear, "xav_uni", "relu")

        self.bn = nn.BatchNorm1d(size)

    def forward(self, x):
        _x = self.bn(self.linear(x))
        _x = x + _x
        _x = F.relu(_x)

        return _x


class SkipForwardNet(nn.Module):
    """
    A skip connection based Feed Forward Neural Network.
    """

    def __init__(self,
                 cfg,
                 skip_block,
                 activation="relu",
                 weight_init="xav_uni"):
        super(SkipForwardNet, self).__init__()
        self.activation = activation
        self.weight_init = weight_init
        self.cfg = copy.deepcopy(cfg)

        in_dim = self.cfg.pop(0)
        out_dim = self.cfg.pop()
        self.linear1 = nn.Linear(in_dim, self.cfg[0])
        self.bn0 = nn.BatchNorm1d(in_dim)
        self.bn1 = nn.BatchNorm1d(self.cfg[0])

        self.blocks = nn.ModuleList()
        for i in range(len(self.cfg)):
            self.blocks.append(skip_block(self.cfg[i]))
            if i != len(self.cfg) - 1 and self.cfg[i] != self.cfg[i + 1]:
                self.blocks.append(nn.Linear(self.cfg[i], self.cfg[i + 1]))
                self.blocks.append(nn.BatchNorm1d(self.cfg[i + 1]))
                self.blocks.append(nn.ReLU())

        self.linear2 = nn.Linear(self.cfg[-1], out_dim)

    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(self.bn1(self.linear1(x)))
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.linear2(x))

        return x


class EarlyStopping:
    """
    Class to help in early stopping experiment. Looks at the incoming
    loss and decides to stop running the experiment when the loss is
    not improving from "patience" steps

    Attributes
    ----------
    config : dict
        Configuration containing the hyperparams
    exp_id : str
        Path to save models weights
    patience_max : int
        Epochs to wait for the loss to improve as compared to the "epoch loss
    patience_step : int
        Epochs since the loss improved
    best_loss : float
        Loss
    best_loss_epoch : int
        Epoch index for which the input loss was minimum

    Methods
    -------
    step(epoch, loss)
        Compare the input loss against best loss and update it if necessary
    """

    def __init__(self, config, patience_max=7):
        self.config = config
        self.exp_id = None
        self.patience_max = patience_max
        self.patience_step = 0
        self.best_loss = None
        self.best_loss_epoch = None

    def step(self, epoch, loss, models):
        """
        Compare the input loss against best loss and update it if necessary

        Parameters
        ----------
        epoch : int
            Epoch index
        loss : float
            Input loss to check against best_loss
        models : tuple
            Tuple containing the models to be used for the experiment

        Returns
        -------
        should_stop : bool
            Indicator to stop the experiment
        best_loss_epoch : int
            The epoch index corresponding to the best loss
        """
        should_stop = False
        if self.best_loss is None:
            self.best_loss = loss
            self.best_loss_epoch = epoch
            save_models(self.config, self.exp_id, models)
        elif self.best_loss > loss:
            self.best_loss = loss
            self.best_loss_epoch = epoch
            self.patience_step = 0
            save_models(self.config, self.exp_id, models)
        elif self.best_loss <= loss:
            self.patience_step += 1
            if self.patience_step == self.patience_max:
                should_stop = True

        return should_stop, self.best_loss_epoch


def init_layer_weights(layer, scheme="xav_uni", activation="relu"):
    """
    Initialize weights of the layer

    Parameters
    ----------
    layer : Tensor
        Weight matrix of the layer
    scheme : string
        Weight supervised scheme to be used
    activation : string
        Activation used in the layer

    """

    if activation != "relu" and activation != "leaky_relu":
        assert "he" not in scheme, "He init can only be used with (relu/leaky_relu)"

    x = {
        "he_nor": nn.init.kaiming_normal_(layer.weight, nonlinearity=activation),
        "he_uni": nn.init.kaiming_uniform_(layer.weight, nonlinearity=activation),
        "xav_nor": nn.init.xavier_normal_(layer.weight),
        "xav_uni": nn.init.xavier_uniform_(layer.weight)
    }.get(scheme, None)
    assert x is not None, "Unknown supervised scheme"

    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def prepare_data(batch, config, models, device):
    """
    Prepare data for the model based on the input arguments. It makes the following
    important decisions during this process:
    1. Should we use the hand-crafted features from scenario?
    2. Should we use the scenario? (Y/N)
        2.1. Should we use the scenario as it is (raw)?
        2.2. Should we create set-embedding of scenario using DeepSet?

    Parameters
    ----------
    batch : Tensor
        Tensor containing the mini-batch data
    config : dict
        Configuration containing the hyperparams
    models : tuple
        Tuple containing the models to be used for the experiment
    device : torch.device
        Device (GPU/CPU) on which to run your models

    Returns
    -------
    x : Tensor
        Model input features
    y : Tensor
        Model output
    """
    c_f, c_v, scenario = batch['c_f'].float().to(device), batch['c_v'].float().to(device), \
                         batch['scenario'].float().to(device)
    bs, num_scenario, num_facility = scenario.shape
    if config['use_feature']:
        features = batch['features'].float().to(device)
        x = T.cat((c_f, c_v, features), dim=1)
    else:
        x = T.cat((c_f, c_v), dim=1)

    y = batch['y'].float().to(device)

    if config['use_scenario'] and config['use_deepset']:
        _, phi, rho = models
        scenario = scenario.reshape(-1, num_facility)
        latent_scenario_1 = phi(scenario)
        latent_scenario_1 = latent_scenario_1.reshape(bs, num_scenario, -1)
        latent_scenario_2 = T.sum(latent_scenario_1, dim=1)
        z = rho(latent_scenario_2)
        x = T.cat((x, z), dim=1)
    elif config['use_scenario'] and not config['use_deepset']:
        z = scenario.reshape(x.shape[0], -1)
        # Mask values from original scenario
        if config['mask_scenario'] > 0:
            mask_percentage = min(1, config['mask_scenario'])
            mask = T.empty_like(z).uniform_() > mask_percentage
            z = z * mask.float().to(device)
            z = z / (1 - mask_percentage)
        x = T.cat((x, z), dim=1)

    return x, y


def zero_grad(config, optimizers):
    """
    Clear the gradient of parameters of models

    Parameters
    ----------
    config : dict
        Configuration containing the hyperparams
    optimizers : tuple
        Tuple containing the optimizers to be used for the experiment
    """
    if config['use_scenario'] and config['use_deepset']:
        opt_model, opt_phi, opt_rho = optimizers
        opt_phi.zero_grad(), opt_rho.zero_grad()
    else:
        opt_model = optimizers
    opt_model.zero_grad()


def step(config, optimizers):
    """
    Update the models using gradients

    Parameters
    ----------
    config : dict
        Configuration containing the hyperparams
    optimizers : tuple
        Tuple containing the optimizers to be used for the experiment
    """
    if config['use_scenario'] and config['use_deepset']:
        opt_model, opt_phi, opt_rho = optimizers
        opt_phi.step(), opt_rho.step()
    else:
        opt_model = optimizers
    opt_model.step()


def _set_model_input_output(config):
    """ Set the model's input and output layer

    Parameters
    ----------
    config: dict
        Configuration containing the hyperparams
    """
    # 10 for c_f and 10 for c_v
    if type(config['model_cfg']) == str:
        model_input_size = 20
        # 170 extracted feature size
        model_input_size = model_input_size + 170 if config['use_feature'] else model_input_size
        if config['use_scenario']:
            model_input_size = model_input_size + config['rho_cfg'][-1] if config[
                'use_deepset'] else model_input_size + 500
        config['model_cfg'] = ast.literal_eval(config['model_cfg'])
        config['model_cfg'].insert(0, model_input_size)
        config['model_cfg'].append(10)


def create_models_and_optimizers(config, device):
    """
    Create models and optimizers to be used in the experiment based on
    the input arguments

    Parameters
    ----------
    config: dict
        Configuration containing the hyperparams
    device : torch.device
        Device (GPU/CPU) on which to run your models

    Returns
    -------
    models : tuple
        Tuple containing the models to be used for the experiment
    optimizers : tuple
        Tuple containing the optimizers to be used for the experiment
    """
    _set_model_input_output(config)

    if config['use_scenario'] and config['use_deepset']:
        config['phi_cfg'] = ast.literal_eval(config['phi_cfg'])
        phi = FeedForwardNet(config['phi_cfg'],
                             dp=config['phi_dp'],
                             activation=config['phi_act'],
                             weight_init=config['weight_init']).to(device)
        opt_phi = optim.SGD(phi.parameters(), lr=config['phi_lr'], weight_decay=config['phi_weight_decay'],
                            momentum=0.9)

        config['rho_cfg'] = ast.literal_eval(config['rho_cfg'])
        rho = FeedForwardNet(config['rho_cfg'],
                             dp=config['rho_dp'],
                             activation=config['rho_act'],
                             weight_init=config['weight_init']).to(device)
        opt_rho = optim.SGD(rho.parameters(), lr=config['rho_lr'], weight_decay=config['rho_weight_decay'],
                            momentum=0.9)

    # Create model and optimizers
    if config['model_type'] == "sfnn":
        model = SkipForwardNet(config['model_cfg'], SkipBlock)
    else:
        model = FeedForwardNet(config['model_cfg'],
                               dp=config['model_dp'],
                               activation=config['model_act'],
                               weight_init=config['weight_init']).to(device)
    opt_model = optim.SGD(model.parameters(), lr=config['model_lr'],
                          momentum=0.9,
                          weight_decay=config['model_weight_decay'])

    if config['use_scenario'] and config['use_deepset']:
        models = (model, phi, rho)
        optimizers = (opt_model, opt_phi, opt_rho)
    else:
        models = (model)
        optimizers = (opt_model)

    return models, optimizers


def save_models(config, exp_id, models):
    """
    Save the models

    Parameters
    ----------
    config: dict
        Configuration containing the hyperparams
    exp_id : str
        Path to save models weights
    models : tuple
        Tuple containing the models to be used for the experiment
    """
    exp_id = str(exp_id)
    if config['use_scenario'] and config['use_deepset']:
        model, phi, rho = models
        T.save(phi.state_dict(), os.path.join(exp_id, 'phi.pt'))
        T.save(rho.state_dict(), os.path.join(exp_id, 'rho.pt'))
    else:
        model = models
    T.save(model.state_dict(), os.path.join(exp_id, 'model.pt'))


def load_models_state_dict(config, exp_id, models):
    """
    Load the models with weight

    Parameters
    ----------
    config : dict
        Configuration containing the hyperparams
    exp_id : str
        Path to the models weights
    models : tuple
        Tuple containing the models to be used for the experiment
    """
    if config['use_scenario'] and config['use_deepset']:
        model, phi, rho = models
        phi.load_state_dict(T.load(os.path.join(exp_id, 'phi.pt')))
        phi.eval()
        rho.load_state_dict(T.load(os.path.join(exp_id, 'rho.pt')))
        rho.eval()
    else:
        model = models
    model.load_state_dict(T.load(os.path.join(exp_id, 'model.pt')))
    model.eval()
