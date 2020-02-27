from itertools import product

from ..utils import const


class SklearnConfigCreator:
    def __call__(self):
        model_configs = {
            const.LINEAR: [{}]
        }
        # model_configs = {
        #     const.LINEAR: [{}],
        #     const.LASSO: [{"alpha": 1}, {"alpha": 0.1}, {"alpha": 10}],
        #     const.RIDGE: [{"alpha": 1}, {"alpha": 0.1}, {"alpha": 10}],
        #     const.ELASTIC: [{}],
        #     const.DECISION_TREE: [{"max_depth": 5, "max_features": "sqrt"},
        #                           {"max_depth": 10, "max_features": "sqrt"},
        #                           {"max_depth": 15, "max_features": "sqrt"}]
        # }

        data_configs = [{const.FEATURE: True, const.SCENARIO: False}]

        # data_configs = [{const.FEATURE: True, const.SCENARIO: False},
        #                 {const.FEATURE: False, const.SCENARIO: True},
        #                 {const.FEATURE: True, const.SCENARIO: True}]

        """Create configurations for SkLearn based models"""
        final_configs = []
        for d_cfg in data_configs:
            for m, m_cfgs in model_configs.items():
                final_conf = {}
                for m_cfg in m_cfgs:
                    final_conf.update({"name": m, "type": const.SKLEARN})
                    final_conf.update(m_cfg)
                    final_conf.update(d_cfg)
                    final_configs.append(final_conf)

        return final_configs


class PyTorchConfigCreator:
    def __call__(self):
        # Only input the hidden layer sizes e.g. "[128]" will create a model of one hidden layer with size 128.
        model_cfg = ["[128, 128]"]
        model_type = ["ffnn"]  # Options: "sfnn" | "ffnn"
        model_lr = [0.001]
        model_dp = [0]
        model_act = ["relu"]
        model_weight_decay = [0]

        # Phi
        phi_cfg = ["[10, 64, 256]"]
        phi_lr = [1e-3]
        phi_dp = [0.0]
        phi_act = ["relu"]
        phi_weight_decay = [0]

        # Rho
        rho_cfg = ["[256, 128]"]
        rho_lr = [1e-3]
        rho_dp = [0.1]
        rho_act = ["relu"]
        rho_weight_decay = [0]

        # Experiment
        num_epochs = [250]
        bs = [256]
        criterion = ["mse"]
        weight_init = ["xav_uni"]
        use_scenario = [0]
        mask_scenario = [0]
        use_deepset = [0]
        use_features = [1]
        use_comet = [0]
        early_stop = [1]

        cfgs_tuples = product(model_cfg, model_type, model_lr, model_dp, model_act, model_weight_decay,
                              phi_cfg, phi_lr, phi_dp, phi_act, phi_weight_decay,
                              rho_cfg, rho_lr, rho_dp, rho_act, rho_weight_decay,
                              num_epochs, bs, criterion, weight_init, use_scenario,
                              mask_scenario, use_deepset, use_features, use_comet, early_stop)

        cfgs = []
        for cfg_tuple in cfgs_tuples:
            mdl_cfg, mdl_type, mdl_lr, mdl_dp, mdl_act, mdl_wd, \
            p_cfg, p_lr, p_dp, p_act, p_wd, \
            r_cfg, r_lr, r_dp, r_act, r_wd, \
            n_ep, b, crit, wi, u_scen, m_scen, \
            u_dpst, u_feat, u_cmt, erly_stp = cfg_tuple

            cfg = {'model_cfg': mdl_cfg,
                   'model_type': mdl_type,
                   'model_lr': mdl_lr,
                   'model_dp': mdl_dp,
                   'model_act': mdl_act,
                   'model_weight_decay': mdl_wd,
                   'phi_cfg': p_cfg,
                   'phi_lr': p_lr,
                   'phi_dp': p_dp,
                   'phi_act': p_act,
                   'phi_weight_decay': p_wd,
                   'rho_cfg': r_cfg,
                   'rho_lr': r_lr,
                   'rho_dp': r_dp,
                   'rho_act': r_act,
                   'rho_weight_decay': r_wd,
                   'num_epochs': n_ep,
                   'bs': b,
                   'criterion': crit,
                   'weight_init': wi,
                   'use_scenario': u_scen,
                   'mask_scenario': m_scen,
                   'use_deepset': u_dpst,
                   'use_feature': u_feat,
                   'use_comet': u_cmt,
                   'early_stop': erly_stp,
                   'data_dir': "data",
                   'type': const.PYTORCH}

            cfgs.append(cfg)

        return cfgs


class ConfigCreator:
    sklearn_config_creator = SklearnConfigCreator()
    sklearn_configs = sklearn_config_creator()

    pytorch_config_creator = PyTorchConfigCreator()
    pytorch_configs = pytorch_config_creator()

    configs = sklearn_configs + pytorch_configs
    # configs = sklearn_configs
    # configs = pytorch_configs