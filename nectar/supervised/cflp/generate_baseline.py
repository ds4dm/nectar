import copy
import random

import numpy as np

from ...utils import save_pickle, load_pickle
from ...utils.combinatorics.cflp import CFLP

random.seed(0)


def solve_surrogate_and_extract_result(cflp):
    result = cflp.solve_two_sip(use_xi_bar=True, gap=0.001, time_limit=300)
    result["obj_val_baseline"] = cflp.evaluate_x(result)

    return result


def generate_baseline(path):
    x_test = np.load(path["data"] / "x_test_raw.npy", allow_pickle=True)
    instance = load_pickle(path["data"] / "instances.pkl")

    baseline = {}
    for item in x_test:
        pid = item["pid"]
        baseline[pid] = {}
        instance[pid].update(instance[-1])
        cflp = CFLP(instance[pid])

        # Average
        cflp.set_xi_bar(np.mean(instance[pid]["scenario"], axis=0))
        baseline[pid]["avg"] = solve_surrogate_and_extract_result(cflp)

        # Randomly sampled one scenario
        random_idx = random.randint(0, cflp.num_scenarios - 1)
        cflp.set_xi_bar(copy.deepcopy(instance[pid]["scenario"][random_idx]))
        baseline[pid]["random_scen"] = solve_surrogate_and_extract_result(cflp)

        # Sample scenario from xi star distribution
        cflp.set_xi_bar(np.random.uniform(20, 40, 10))
        baseline[pid]["random_dist"] = solve_surrogate_and_extract_result(cflp)

    save_pickle(path["data"] / "baseline.pkl", baseline)
