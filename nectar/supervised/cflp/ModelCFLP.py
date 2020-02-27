from nectar.utils.combinatorics.cflp import CFLP
from ..Model import Model
from ...utils import load_pickle, save_pickle


class ModelCFLP(Model):
    def __init__(self, config, path):
        super(ModelCFLP, self).__init__(config, path)

    @staticmethod
    def evaluate_optimization_metrics(exp_path):
        y_test_pred = load_pickle(exp_path / "y_test_pred_raw.pkl")
        instance = load_pickle(Model.path["instance"])

        result_xi_pred = {}
        for item in Model.x_test_raw:
            pid = item["pid"]
            instance[pid].update(instance[-1])
            cflp = CFLP(instance[pid])
            cflp.set_xi_bar(y_test_pred[pid])
            result_xi_pred[pid] = cflp.solve_two_sip(use_xi_bar=True, gap=0.001, time_limit=300)
            result_xi_pred[pid]["obj_val_pred"] = cflp.evaluate_x(result_xi_pred[pid])
            result_xi_pred[pid]["xi_hat"] = y_test_pred[pid]

        save_pickle(exp_path / "result_xi_pred.pkl", result_xi_pred)
