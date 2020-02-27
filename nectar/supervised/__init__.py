from ..utils import load_pickle
from ..utils import const


def find_best_experiments(experiments_path):
    result = []
    best_exp = {'loss_sklearn': None,
                'loss_pytorch': None,
                'sklearn_path': None,
                'pytorch_path': None}

    for p in experiments_path.iterdir():
        config = load_pickle(p.joinpath(const.CONFIG_FILE))
        metrics = load_pickle(p.joinpath(const.LEARNING_METRIC_FILE))
        if config['type'] == const.SKLEARN:
            if best_exp['loss_sklearn'] and (best_exp['loss_sklearn'] < metrics['test_loss']):
                best_exp['loss_sklearn'] = metrics['test_loss']
                best_exp['sklearn_path'] = p
            else:
                best_exp['loss_sklearn'] = metrics['test_loss']
                best_exp['sklearn_path'] = p
        elif config['type'] == const.PYTORCH:
            if best_exp['loss_pytorch'] and \
                    (best_exp['loss_pytorch'] < metrics['test_loss']):
                best_exp['loss_pytorch'] = metrics['test_loss']
                best_exp['pytorch_path'] = p
            else:
                best_exp['loss_pytorch'] = metrics['test_loss']
                best_exp['pytorch_path'] = p

    best_exp['sklearn_path'] and result.append(best_exp['sklearn_path'])
    best_exp['pytorch_path'] and result.append(best_exp['pytorch_path'])

    return result