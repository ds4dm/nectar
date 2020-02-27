"""
Module to generate optimal solution (with some gap) for the instances.

Runs n_worker parallel processes, each solving an instance[i]. Once the workers
have finished running, the result of individual instances are collated into a
dictionary indexed by the instance id and saved.
"""
import multiprocessing as mp

from ...utils import save_pickle, load_pickle
from ...utils.combinatorics.cflp import CFLP


def worker_solve_optimally(wid,
                           instance,
                           optimality_gap,
                           time_limit,
                           shared_result_ext,
                           q,
                           lock):
    """Worker to solve extensive S-CFLP

    Parameters
    ----------
    wid : int
        Worker id
    instance : defaultdict(dict)
        Dictionary containing problem instances
    optimality_gap : float, between [0, 1]
        Maximum gap allowed in percent from optimal objective value
    time_limit : int
        Maximum time allowed to solve the instance
    shared_result_ext : mp.managers.DictProxy
        Shared dictionary containing the extensive result
    q : mp.queues.Queue
        Shared queue containing the id's of problem to solve
    lock : mp.synchronize.Lock
        Lock to access the shared queue. Ensures that only one
        worker is able to pop from the queue.
    """
    print(f"Starting worker {wid} ...")

    while True:
        with lock:
            if q.empty():
                print(f"Worker {wid} finished.")
                return
            pid = q.get()

        if pid in shared_result_ext and (shared_result_ext[pid]['gap'] <= optimality_gap):
            print(f"Instance {pid} is already solved with {optimality_gap} gap")
            continue

        instance[pid].update(instance[-1])
        cflp = CFLP(instance[pid])
        result = cflp.solve_two_sip(use_xi_bar=False, gap=optimality_gap, time_limit=time_limit)
        shared_result_ext[pid] = result
        print(f'W {wid} solved problem {pid} in {result["run_time"]:.2f}s'
              f' with objective {result["obj_val"]:.2f}')


def solve_optimally(instance,
                    result_ext_file_path,
                    n_worker,
                    optimality_gap,
                    time_limit,
                    from_pid,
                    to_pid):
    """Parent to solve extensive S-CFLP instances in parallel

    Parameters
    ----------
    instance : defaultdict(dict)
        Dictionary containing problem instances
    result_ext_file_path : pathlib.Path
        Result extensive file path
    n_worker : int
        Number of worker to run in parallel
    optimality_gap : float between [0, 1]
        Maximum gap allowed in percent from optimal objective value
    time_limit : int
        Time allowed in seconds to solve the extensive form
    from_pid : int
        Id of first instance to solve
    to_pid : int
        Id of last instance to solve
    """
    # Create lock and queue with problem ids to solve extensively
    lock = mp.Lock()
    q = mp.Queue()
    manager = mp.Manager()
    shared_result_ext = manager.dict()
    if result_ext_file_path.exists():
        result_ext = load_pickle(result_ext_file_path, check=False)
        for k, v in result_ext.items():
            shared_result_ext[k] = v

    # Solve extensive form of problem
    [q.put(i) for i in range(from_pid, to_pid)]
    workers = []
    for rank in range(n_worker):
        p = mp.Process(target=worker_solve_optimally,
                       args=(rank,
                             instance,
                             optimality_gap,
                             time_limit,
                             shared_result_ext,
                             q,
                             lock))
        workers.append(p)
    [w.start() for w in workers]
    [w.join() for w in workers]

    result_ext = dict(shared_result_ext)
    save_pickle(result_ext_file_path, result_ext)


def generate_optimal_sol(meta_config, problem_config, path):
    """Generate optimal solution

    Parameters
    ----------
    meta_config : configparser.ConfigParser
        Project configuration
    problem_config : configparser.ConfigParser
        Problem Configuration
    path : dict
        Dictionary of importlib.Path objects
    """
    n_worker = meta_config.getint('Run', 'n_worker')
    from_pid = meta_config.getint('Run', 'from_pid')
    to_pid = meta_config.getint('Run', 'to_pid')

    optimality_gap = problem_config.getfloat('Problem', 'extensive_optimality_gap')
    time_limit = problem_config.getint('Problem', 'extensive_time_limit')

    instance = load_pickle(path["instance"])
    solve_optimally(instance,
                    path["result_ext"],
                    n_worker,
                    optimality_gap,
                    time_limit,
                    from_pid,
                    to_pid)
