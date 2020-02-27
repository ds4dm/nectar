import multiprocessing as mp
from operator import itemgetter

import numpy as np

from ...utils import save_pickle, load_pickle
from ...utils.combinatorics.cflp import CFLP


def prune_xi_hat(result_xi, max_xi_hat):
    """Prune xi_hat to 0.9 if its gets below 1 or to max_xi_hat + 1
    if it gets above max_xi_hat

    In the next iteration to update the representative scenario, we
    check if xi_hat is between 1 and max_xi_hat. Since we modified
    the xi_hat to be either 0.9 (underflow) or max_xi_hat + 1 (overflow),
    we will not pick this location for modification.

    Parameters
    ----------
    result_xi : dict
        Dict containing result xi
    max_xi_hat : float
        Maximum demand allowed at a client node in representative
        scenario
    """
    xi_hat = result_xi['xi_hat']
    # We are adding 1 to max_xi_hat and replacing by 0.9 so that they get out of bound
    # and are not picked in the next iteration
    for idx in range(len(xi_hat)):
        xi_hat[idx] = max_xi_hat + 1 if xi_hat[idx] > max_xi_hat else xi_hat[idx]
        xi_hat[idx] = 0 if xi_hat[idx] < 0 else xi_hat[idx]


def mask_demand_at_facility_open_only_in_xi_hat(result_ext, result_surr, xi_hat):
    """
    Zero out demand at all the facility in the representative scenario, which
    are open according to the surrogate (problem with only representative scenario)
    result but closed according to the extensive result.

    Parameters
    ----------
    result_ext : dict
        Dict containing result extensive
    result_surr : dict
        Dict containing result surrogate
    xi_hat : np.ndarray
        Representative demand scenario
    """
    # Check that the same nodes are open
    facility_open_only_in_xi = result_ext["b"] - result_surr["b"]
    facility_open_only_in_xi = [True if item == -1 else False for item in facility_open_only_in_xi]
    xi_hat[facility_open_only_in_xi] = 0


def get_max_diff_ordered_facilities(diff):
    """
    Get the facilities sorted by the absolute maximum difference between capacity
    installed in extensive and surrogate solution

    Parameters
    ----------
    diff : array
         Array containing difference between capacity in the
         extensive and surrogate result

    Returns
    -------
    ordered_facilities : list
        List of facilities sorted by absolute maximum difference between capacity
        installed in extensive and surrogate solution
    """
    diff_abs = np.abs(diff)
    diff_abs = [[diff_item, idx] for idx, diff_item in enumerate(diff_abs)]
    diff_abs = sorted(diff_abs, key=itemgetter(0))
    ordered_facilities = [diff_abs_item[1] for diff_abs_item in diff_abs]
    ordered_facilities.reverse()

    return ordered_facilities


def get_first_ordered_facility_with_bounded_demand_in_xi_hat(ordered_facilities,
                                                             xi_hat,
                                                             max_xi_hat):
    for idx in ordered_facilities:
        if 1 <= xi_hat[idx] <= max_xi_hat:
            return idx

    return -1


def update_by_diff_fraction(xi_hat, diff, facility_idx, fraction=0.5):
    """
    Update xi_hat by fraction of difference between capacity installed
    in result_ext and result_surr at facility_idx

    Parameters
    ----------
    xi_hat : np.ndarray
         Representative demand scenario
    diff : array
         Array containing difference between capacity in the
         extensive and surrogate result
    facility_idx : int
        Index of the facility to change the demand
    fraction :

    Returns
    -------

    """
    assert 0 <= fraction <= 1, "Fraction should be between [0, 1]"
    assert xi_hat.shape == diff.shape, "Xi hat and diff should be of same shape"
    assert 0 <= facility_idx <= len(xi_hat) - 1

    xi_hat[facility_idx] += (diff[facility_idx] * 0.5)


def update_by_percent(xi_hat, diff, facility_idx, percent=0.05):
    """
    Update xi_hat by input percent of its current value.

    If the diff vector is entirely positive or negative, we update
    the entire xi_hat vector accordingly. If not then we only change
    the demand at facility_idx.

    Parameters
    ----------
    xi_hat : array
         Representative demand scenario
    diff : array
         Array containing difference between capacity in the
         extensive and surrogate result
    facility_idx : int
        Index of the facility to change the demand
    percent : float [0, 1]
        Percentage by which to update xi_hat
    """
    assert 0 <= percent <= 1, "Percentage should be between [0, 1]"
    assert xi_hat.shape == diff.shape, "Xi hat and diff should be of same shape"
    assert 0 <= facility_idx <= len(xi_hat) - 1

    flag = False
    if (diff >= 0).all():
        # print("All positive")
        xi_hat *= (1 + percent)
        flag = True
    elif (diff <= 0).all():
        # print("All negative")
        xi_hat *= (1 - percent)
        flag = True

    if not flag:
        if diff[facility_idx] > 0:
            xi_hat[facility_idx] *= (1 + percent)
        elif diff[facility_idx] < 0:
            xi_hat[facility_idx] *= (1 - percent)


def worker_generate_xi_hat(wid,
                           instance,
                           result_ext,
                           heuristic,
                           shared_result_xi,
                           q,
                           lock,
                           threshold_obj=1):
    print(f"Starting worker {wid}")
    while True:
        # Set iterators
        mask_demand = heuristic.getint('mask_demand')
        average_reset = heuristic.getint('average_reset')
        average_reset_fraction = heuristic.getfloat('average_reset_fraction')
        linear_decrease = heuristic.getint('linear_decrease')
        linear_decrease_from = heuristic.getint('linear_decrease_from')
        linear_decrease_to = heuristic.getint('linear_decrease_to')
        custom_decrease = heuristic.getint('custom_decrease')

        if linear_decrease:
            linear_step_size = (linear_decrease_from - linear_decrease_to) / linear_decrease

        # Get the problem-id to solve
        with lock:
            if q.empty():
                print(f"Worker {wid} finished.")
                return
            pid = q.get()

        instance[pid].update(instance[-1])  # Update instance with c_tv, c_tf
        cflp = CFLP(instance[pid])

        xi_hat = np.average(instance[pid]['scenario'], axis=0)
        max_xi_hat = np.max(np.sum(instance[pid]['scenario'], axis=1))
        result_xi = cflp.initialize_result_xi(heuristic, xi_hat)

        obj_val_bound = result_ext[pid]["obj_val"] * (1 + (threshold_obj / 100))

        should_stop = False
        iteration = 0

        while should_stop is False:
            result_xi['solved_surr'] = False
            cflp.set_xi_bar(result_xi['xi_hat'])
            result_surr = cflp.solve_two_sip(use_xi_bar=True, gap=0.001, time_limit=300)
            if result_surr['gap'] <= 0.001:
                result_xi['solved_surr'] = True
            else:
                # if result_surr['gap'] > 0.001:
                print(f"W {wid} P {pid}: Surrogate unable to close gap at iteration {iteration}")
                break

            is_bounded = False
            obj = cflp.evaluate_x(result_surr)
            if obj <= obj_val_bound:
                result_xi["b"] = result_surr["b"]
                result_xi["v"] = result_surr["v"]
                result_xi["obj_val"] = obj
                result_xi["solved_xi"] = True
                is_bounded = True

            if is_bounded:
                print(f'W {wid} P {pid}: Xi found in {iteration}, '
                      f'{result_ext[pid]["obj_val"]}, {obj_val_bound}, {obj}')
                break

            # Update xi_hat
            if mask_demand:
                mask_demand_at_facility_open_only_in_xi_hat(result_ext[pid],
                                                            result_surr,
                                                            result_xi['xi_hat'])
                mask_demand -= 1
            else:
                diff = result_ext[pid]["v"] - result_surr["v"]
                ordered_facilities = get_max_diff_ordered_facilities(diff)
                facility_idx = \
                    get_first_ordered_facility_with_bounded_demand_in_xi_hat(ordered_facilities,
                                                                             result_xi['xi_hat'],
                                                                             max_xi_hat)
                if facility_idx < 0:
                    print(f'W {wid} P {pid}: Out of bound xi {iteration},{result_ext[pid]["obj_val"]}'
                          f', {obj_val_bound}, {obj}')
                    break
                if diff[facility_idx] == 0:
                    print(f'W {wid} P {pid}: No difference in v_opt and v_surr {iteration}, '
                          f'{result_ext[pid]["obj_val"]}, {obj_val_bound}, {obj}')
                    break
                if average_reset:
                    update_by_diff_fraction(result_xi['xi_hat'],
                                            diff,
                                            facility_idx,
                                            fraction=average_reset_fraction)
                    average_reset -= 1
                elif linear_decrease:
                    percent = ((linear_decrease * linear_step_size) + linear_decrease_to) / 100
                    # print(percent)
                    update_by_percent(result_xi['xi_hat'], diff, facility_idx, percent)
                    linear_decrease -= 1
                else:
                    should_stop = True
                prune_xi_hat(result_xi, max_xi_hat)

            iteration += 1

        result_xi['exit_iteration'] = iteration
        shared_result_xi[pid] = result_xi
        if should_stop:
            print(f'W {wid} P {pid}: Exhausted iterations {result_ext[pid]["obj_val"]}, '
                  f'{obj_val_bound}, {obj}')


def generate_xi_hat(meta_config, heuristic, path):
    """Generate xi_hat by heuristics

    Parameters
    ----------
    meta_config : configparser.ConfigParser
        Project configuration
    heuristic : configparser.SectionProxy
        Heuristic to use to generate_xi_hat
    path : dict
        Dictionary containing importlib.Path objects
    """
    n_worker = meta_config.getint('Run', 'n_worker')
    from_pid = meta_config.getint('Run', 'from_pid')
    to_pid = meta_config.getint('Run', 'to_pid')

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
        if i in result_ext and (i not in result_xi or
                                i in result_xi and result_xi[i]['solved_xi'] is False):
            q.put(i)
        else:
            print(f"Unsolved ext or solved xi for {i}")

    workers = []
    for rank in range(n_worker):
        p = mp.Process(target=worker_generate_xi_hat,
                       args=(rank, instance, result_ext, heuristic, shared_result_xi, q, lock))
        workers.append(p)
    [w.start() for w in workers]
    [w.join() for w in workers]

    result_xi = dict(shared_result_xi)
    save_pickle(path["result_xi"], result_xi)
