import copy
import multiprocessing as mp

import numpy as np

from ...utils import load_pickle
from ...utils import save_pickle
from ...utils.combinatorics.cflp import CFLP


def worker_improve_xi_hat(wid, instance, result_ext, shared_result_xi, q, lock):
    """Worker to improve xi_hat

    Parameters
    ----------
    wid : int
        Worker id
    instance : defaultdict(dict)
        Dictionary containing problem instances
    result_ext : dict
        Dict containing result extensive
    shared_result_xi :
        Shared result xi
    q : mp.queues.Queue
        Shared queue containing the id's of problem to solve
    lock : mp.synchronize.Lock
        Lock to access the shared queue. Ensures that only one
        worker is able to pop from the queue.
    """
    print(f"Starting worker {wid}")

    while True:
        with lock:
            if q.empty():
                print(f"Worker {wid} finished.")
                return
            pid = q.get()

        instance[pid].update(instance[-1])
        cflp = CFLP(instance[pid])
        obj_val = cflp.evaluate_x(shared_result_xi[pid])

        # Find ratio of total capacity installed in optimal and surrogate solution
        xi_hat_scaling_factor = np.sum(result_ext[pid]['v']) / np.sum(shared_result_xi[pid]['v'])
        # Scale xi_hat with ratio found in last block and evaluate it
        xi_hat_scaled = xi_hat_scaling_factor * shared_result_xi[pid]['xi_hat']
        cflp.set_xi_bar(xi_hat_scaled)
        result_surr_scaled = cflp.solve_two_sip(use_xi_bar=True, gap=0.001, time_limit=300)
        obj_val_scaled = cflp.evaluate_x(result_surr_scaled)

        if obj_val_scaled < obj_val:
            # Update shared_result_xi
            shared_result_xi[pid]['prev_xi_hat'] = copy.deepcopy(shared_result_xi[pid]['xi_hat'])
            shared_result_xi[pid]['xi_hat'] = xi_hat_scaled
            shared_result_xi[pid]['prev_obj_val'] = copy.deepcopy(shared_result_xi[pid]['obj_val'])
            shared_result_xi[pid]['obj_val'] = obj_val_scaled
            shared_result_xi[pid]['is_improved'] = True

            print(f'W {wid} P {pid} Old obj: {obj_val}, New obj: {obj_val_scaled}')
        else:
            print(f"W {wid} P {pid} xi_hat not improved {obj_val} {obj_val_scaled}")


def improve_xi_hat(exp_config, path):
    """Improve xi_hat by altering the representative demand scenario

    Parameters
    ----------
    meta_config : configparser.ConfigParser
        Project configuration
    path : dict
        Dictionary of importlib.Path objects
    """
    n_worker = exp_config.getint('Run', 'n_worker')
    from_pid = exp_config.getint('Run', 'from_pid')
    to_pid = exp_config.getint('Run', 'to_pid')

    instance = load_pickle(path["instance"])
    result_ext = load_pickle(path["result_ext"])
    result_xi = load_pickle(path["result_xi"])

    q = mp.Queue()
    lock = mp.Lock()
    manager = mp.Manager()
    shared_result_xi = manager.dict()
    for k, v in result_xi.items():
        shared_result_xi[k] = v

    for i in range(from_pid, to_pid):
        if i in result_xi and result_xi[i]['solved_xi'] and \
                not result_xi[i]['is_improved']:
            q.put(i)

    workers = []
    for rank in range(n_worker):
        p = mp.Process(target=worker_improve_xi_hat,
                       args=(rank, instance, result_ext, shared_result_xi, q, lock))
        workers.append(p)
    [w.start() for w in workers]
    [w.join() for w in workers]

    result_xi = dict(shared_result_xi)
    save_pickle(path["result_xi"], result_xi)
