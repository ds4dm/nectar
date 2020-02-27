import numpy as np
import pandas as pd

from nectar.utils import load_pickle


def create_row_entry(pid, result, suffix=""):
    row = [pid,
           result["obj_val" + suffix],
           result["obj_time"][-1][-1]]
    row.extend(result["b"])
    row.extend(result["v"])

    return row


def extract_experiment_result(pids, experiment):
    rows = []
    result = load_pickle(experiment / "result_xi_pred.pkl")

    [rows.append(create_row_entry(pid, result[pid], suffix="_pred")) for pid in pids]

    return rows


def extract_result(path, experiments, problem_config):
    title = ["pid", "Objective", "Time"]
    n_facility = problem_config.getint("Problem", "n_facility")
    title.extend(["_".join(["b", str(i)]) for i in range(n_facility)])
    title.extend(["_".join(["v", str(i)]) for i in range(n_facility)])
    x_test_raw = np.load(path["data"] / "x_test_raw.npy", allow_pickle=True)
    pids = [i["pid"] for i in x_test_raw]

    # Fetch baseline results
    result_ext = load_pickle(path["data"] / "result_ext.pkl")
    baseline = load_pickle(path["data"] / "baseline.pkl")
    baseline_avg, baseline_random_scen, baseline_extensive, \
    baseline_random_dist = [], [], [], []

    for pid in pids:
        result = baseline[pid]
        baseline_avg.append(create_row_entry(pid, result["avg"], suffix="_baseline"))
        baseline_random_scen.append(create_row_entry(pid, result["random_scen"], suffix="_baseline"))
        baseline_random_dist.append(create_row_entry(pid, result["random_dist"], suffix="_baseline"))
        result = result_ext[pid]
        baseline_extensive.append(create_row_entry(pid, result))

    pd.DataFrame(data=baseline_avg, columns=title) \
        .to_csv(path["data"] / "baseline_avg.csv")
    pd.DataFrame(data=baseline_random_scen, columns=title) \
        .to_csv(path["data"] / "baseline_random_scen.csv")
    pd.DataFrame(data=baseline_random_dist, columns=title) \
        .to_csv(path["data"] / "baseline_random_dist.csv")
    pd.DataFrame(data=baseline_extensive, columns=title) \
        .to_csv(path["data"] / "baseline_extensive.csv")

    # Fetch best experiment results
    time_preprocessing = np.load(path["data"] / "preprocessing_time.npy")[0]
    for experiment in experiments:
        rows = extract_experiment_result(pids, experiment)
        df = pd.DataFrame(data=rows, columns=title)
        time_prediction = load_pickle(experiment / "time_prediction.pkl")
        df["Time"] += time_preprocessing + time_prediction
        df.to_csv(str(path["data"]) + "/exp_" + str(experiment).split("/")[-1] + ".csv")
