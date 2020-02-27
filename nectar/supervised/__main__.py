from argparse import ArgumentParser
from configparser import ConfigParser
from importlib import import_module
from pathlib import Path

from .configs_creator import ConfigCreator
from ..supervised import find_best_experiments
from ..utils import const
from ..utils import load_pickle


class ModelFactory:
    def __call__(self, problem_path, config, path):
        if config["type"] == const.SKLEARN:
            model = getattr(import_module(".".join([problem_path, "ModelSkLearn"])), "ModelSkLearn")
        elif config['type'] == const.PYTORCH:
            model = getattr(import_module(".".join([problem_path, "ModelPyTorch"])), "ModelPyTorch")

        return model(config, path)


def main():
    model_factory = ModelFactory()
    exp_config, problem_config = ConfigParser(), ConfigParser()
    ROOT = Path(__file__).parent.parent
    # Meta config
    exp_config.read(ROOT / "config" / "meta.ini")
    data_dir = exp_config.get('Directory', 'data')
    problem = exp_config['Run']['problem']
    instance_file = exp_config.get('File', 'instance')
    result_ext_file = exp_config.get('File', 'result_extensive')
    result_xi_file = exp_config.get('File', 'result_xi')
    # Problem config
    problem_config.read(ROOT / "config" / ".".join([exp_config['Run']['problem'], "ini"]))
    problem_path = ".".join(["nectar.supervised", problem])
    get_problem_identifier = getattr(import_module(
        "nectar.utils.combinatorics."+problem
    ), "get_problem_identifier")
    identifier = get_problem_identifier(problem_config)

    data_dir_path = ROOT / data_dir / "_".join([problem, identifier])
    path = {
        "data": data_dir_path,
        "result_xi": data_dir_path / result_xi_file,
        "result_ext": data_dir_path / result_ext_file,
        "instance": data_dir_path / instance_file
    }

    parser = ArgumentParser()
    parser.add_argument('--run', type=str,
                        help='specify the data_manager module to execute. '
                        'train : Train ML model'
                        'eval_opt_metric : Evaluate optimization metric'
                        'baseline : '
                        'extract : ',
                        default='train')
    args = parser.parse_args()
    if args.run == "train":
        for model_config in ConfigCreator.configs:
            model = model_factory(problem_path, model_config, path)
            model.create_experiment_folder()
            model.train()
            model.predict()
            model.evaluate_learning_metrics()
            model.save()
            del model
    elif args.run == "eval_opt_metric":
        experiments = find_best_experiments(data_dir_path / "experiments")
        for experiment in experiments:
            print(f"Evaluating the combinatorial metric for: {experiment}")
            model_config = load_pickle(experiment / "config.pkl")
            model = model_factory(problem_path, model_config, path)
            model.exp_path = experiment
            model.evaluate_optimization_metrics(experiment)
            del model
    elif args.run == "baseline":
        generate_baseline = getattr(import_module(".".join([problem_path,
                                                            "generate_baseline"])),
                                    "generate_baseline")
        generate_baseline(path)
    elif args.run == "extract":
        extract_result = getattr(import_module(".".join([problem_path,
                                                        "extract_result"])),
                                 "extract_result")
        experiments = find_best_experiments(data_dir_path / "experiments")
        extract_result(path, experiments, problem_config)


if __name__ == "__main__":
    main()
