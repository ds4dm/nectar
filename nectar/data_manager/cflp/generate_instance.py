"""
Module to generate instance for Stochastic Capacitated Facility Location Problem

An instance comprises of random first stage cost and fixed (across all the instances)
second stage cost.
"""

from collections import defaultdict

import numpy as np

from ...utils import load_pickle
from ...utils import save_pickle


def generate_first_stage_cost(from_pid, to_pid, n_facility, n_client, n_scenario, cost):
    """Generate fixed cost for opening a facility and variable cost based on
     the capacity installed capacity at a facility.

    Parameters
    ----------
    from_pid : int
        Start seed to create instances
    to_pid : int
        End seed to create instance
    n_facility : int
        Number of facilities
    n_client : int
        Number of clients
    n_scenario : int
        Number of scenarios
    cost : defaultdict(dict)
        Previously generated instances

    Returns
    -------
    cost : defaultdict(dict)
        Generated instances
    """
    cost = defaultdict(dict) if cost is None else cost

    for i in range(from_pid, to_pid):
        if i in cost:
            print(f"Instance {i} already generated")
            continue
        np.random.seed(i)

        c_f = np.random.randint(15, 20, n_facility)
        c_v = np.random.randint(5, 10, n_facility)
        demand_mean = np.floor((c_f + 10 * c_v) / np.sqrt(n_facility))
        scenario = np.array([[np.random.poisson(demand_mean[j])
                              for j in range(n_client)]
                             for _ in range(n_scenario)])

        cost[i] = {'c_f': c_f, 'c_v': c_v, 'scenario': scenario}

    return cost


def generate_second_stage_cost(n_client, n_facility, cost, test=False):
    """Generate fixed cost for serving a client from a facility and variable cost
    based on the demand being served by a facility.

    Parameters
    ----------
    n_client : int
        Number of client
    n_facility : int
        Number of facility
    cost : defaultdict(dict)
        Generated instances (first-stage cost)
    test : bool
        Boolean indicating whether we are running a test

    Returns
    -------
    cost : defaultdict(dict)
        Generated instances (with second-stage cost)
    """
    if -1 not in cost:
        # Generate static c_tv, c_tf
        c_tf = 10 * np.array([[abs(i - j) for j in range(n_client)]
                              for i in range(n_facility)])
        c_tv = np.array([[abs(i - j) for j in range(n_client)]
                         for i in range(n_facility)])

        c_tf_hub = np.zeros(n_facility)
        c_tv_hub = np.zeros(n_client)
        if not test:
            for j in range(n_client):
                c_tf_hub[j] = 5 * np.max(c_tf[:, j])
                c_tv_hub[j] = 5 * np.max(c_tv[:, j])
        else:
            # During test, we set the fixed and variable cost from hub
            # equal to zero. Since there is no limit on hub capacity,
            # the demand for each client should be meet from hub.
            for j in range(n_client):
                c_tf_hub[j] = 0
                c_tv_hub[j] = 0
        c_tf = np.vstack((c_tf_hub.reshape(1, -1), c_tf))
        c_tv = np.vstack((c_tv_hub.reshape(1, -1), c_tv))
        # print(c_tf[10:], c_tv)

        assert c_tf.shape == (n_facility + 1, n_client)
        assert c_tv.shape == (n_facility + 1, n_client)
        assert c_tf[1, 0] == 0 and c_tf[n_facility, n_client - 1] == 0 and c_tf[3, 3] != 0
        assert c_tv[1, 0] == 0 and c_tv[n_facility, n_client - 1] == 0 and c_tv[3, 3] != 0
        # print(c_tf, c_tv)

        cost[-1] = {'c_tf': c_tf, 'c_tv': c_tv}

    return cost


def generate_instance(meta_config, problem_config, path, test=False):
    """Generate problem instances for Stochastic Capacitated Facility Location
    Problem.

    Parameters
    ----------
    meta_config : configparser.ConfigParser
        Project configuration
    problem_config : configparser.ConfigParser
        Problem configuration
    path : dict
        Dictionary of importlib.Path objects
    test : bool (default False)
        Boolean indicating whether we are writing test or not
    """
    from_pid = meta_config.getint('Run', 'from_pid')
    to_pid = meta_config.getint('Run', 'to_pid')

    n_client = problem_config.getint('Problem', 'n_client')
    n_facility = problem_config.getint('Problem', 'n_facility')
    n_scenario = problem_config.getint('Problem', 'n_scenario')

    cost = load_pickle(path["instance"], check=False) if path["instance"].exists() else None
    cost = generate_first_stage_cost(from_pid, to_pid, n_facility, n_client, n_scenario, cost)
    cost = generate_second_stage_cost(n_client, n_facility, cost, test)
    save_pickle(path["instance"], cost)
