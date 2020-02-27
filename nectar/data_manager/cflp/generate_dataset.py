import random
from collections import defaultdict
import time
import numpy as np

from ...utils import load_pickle

np.random.seed(7)
random.seed(11)

MIN_C_F, MAX_C_F = 15, 19
MIN_C_V, MAX_C_V = 5, 9

MEAN_C_F = (MAX_C_F - MIN_C_F) / 2
MEAN_C_V = (MAX_C_V - MIN_C_V) / 2


def fetch_scenario(idxs, data):
    scenario = []
    for idx in idxs:
        scenario.append(data[idx]['scenario'])
    scenario = np.asarray(scenario)

    return scenario


def normalize_scenario(scenario, MIN_SCE, MAX_SCE):
    scenario_diff = np.subtract(scenario, MIN_SCE)
    scenario_scaled = np.divide(scenario_diff, MAX_SCE - MIN_SCE)
    scenario_scaled = (scenario_scaled * 2) - 1

    return scenario_scaled


def extract_scenario_features(scenario):
    features = []
    start_time = time.time()
    features.extend(np.max(scenario, axis=0))
    features.extend(np.min(scenario, axis=0))
    features.extend(np.median(scenario, axis=0))
    features.extend(np.quantile(scenario, 0.75, axis=0))
    features.extend(np.quantile(scenario, 0.25, axis=0))
    features.extend(np.mean(scenario, axis=0))
    features.extend(np.std(scenario, axis=0))

    for k in [0.9, 1, 1.1, 1.2, 1.5]:
        greater_than = []
        less_than = []
        for i in range(scenario.shape[1]):
            i_greater_than = [True] * scenario.shape[0]
            i_less_than = [True] * scenario.shape[0]
            for j in range(scenario.shape[1]):
                if i == j:
                    continue

                i_greater_than = np.logical_and(i_greater_than, (1 + k) * scenario[:, i] >= scenario[:, j])
                i_less_than = np.logical_and(i_less_than, scenario[:, i] <= (1 + k) * scenario[:, j])

            greater_than.append(sum(i_greater_than) / scenario.shape[0])
            less_than.append(sum(i_less_than) / scenario.shape[0])

        features.extend(greater_than)
        features.extend(less_than)

    total_time = time.time() - start_time

    return np.asarray(features), total_time


def create_model_input(idxs, instance, cost_normalized, scenarios_normalized):
    assert len(idxs) == scenarios_normalized.shape[0]
    total_time = 0
    x = []
    for rank, idx in enumerate(idxs):
        x_object = {k: v for k, v in instance[idx].items()}
        x_object["pid"] = idx
        x_object["c_f_normalized"] = cost_normalized[idx]['c_f']
        x_object["c_v_normalized"] = cost_normalized[idx]['c_v']
        x_object["scenario_normalized"] = scenarios_normalized[rank]
        x_object["scenario_features"], item_time = extract_scenario_features(scenarios_normalized[rank])
        total_time += item_time
        x.append(x_object)

    return {"input": np.asarray(x), "total_time": total_time}


def generate_dataset(path, train_test_split=0.7):
    instance = load_pickle(path["instance"])
    result_xi = load_pickle(path["result_xi"])
    total_time = 0

    # Find problem for which we have representative scenario
    solved = []
    for k, v in result_xi.items():
        v["solved_xi"] and solved.append(k)

    # Normalize cost
    cost_normalized = defaultdict(dict)
    start_time = time.time()
    for idx in solved:
        cost_normalized[idx]['c_f'] = (((instance[idx]['c_f'] - MIN_C_F) / (MAX_C_F - MIN_C_F)) * 2) - 1
        cost_normalized[idx]['c_v'] = (((instance[idx]['c_v'] - MIN_C_V) / (MAX_C_V - MIN_C_V)) * 2) - 1
    total_time += (time.time() - start_time)

    # Shuffle and split into train and test
    random.shuffle(solved)
    n_train = int(train_test_split * len(solved))
    train_idxs, test_idxs = solved[:n_train], solved[n_train:]

    # Normalize scenarios
    train_scenarios = fetch_scenario(train_idxs, instance)
    test_scenarios = fetch_scenario(test_idxs, instance)
    start_time = time.time()
    MAX_SCE = np.max(train_scenarios, axis=0)
    MIN_SCE = np.min(train_scenarios, axis=0)
    train_scenarios_normalized = normalize_scenario(train_scenarios, MIN_SCE, MAX_SCE)
    test_scenarios_normalized = normalize_scenario(test_scenarios, MIN_SCE, MAX_SCE)
    total_time += (time.time() - start_time)

    # Prepare training samples
    result = create_model_input(train_idxs, instance, cost_normalized, train_scenarios_normalized)
    x_train, total_time_train = result["input"], result["total_time"]

    result = create_model_input(test_idxs, instance, cost_normalized, test_scenarios_normalized)
    x_test, total_time_test = result["input"], result["total_time"]

    total_time += (total_time_train + total_time_test)

    y_train = np.asarray([{"pid": pid, "xi_hat": result_xi[pid]["xi_hat"]}
                          for pid in train_idxs])
    y_test = np.asarray([{"pid": pid, "xi_hat": result_xi[pid]["xi_hat"]}
                         for pid in test_idxs])

    np.save(path["data"] / "x_train_raw.npy", x_train)
    np.save(path["data"] / "y_train_raw.npy", y_train)
    np.save(path["data"] / "x_test_raw.npy", x_test)
    np.save(path["data"] / "y_test_raw.npy", y_test)
    np.save(path["data"] / "preprocessing_time.npy", [total_time / len(solved)])
