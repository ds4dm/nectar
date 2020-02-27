"""Dataset management

Data manager comprises of following modules:
1. generate_instance.py
2. generate_optimal_sol.py
3. generate_xi_star.py (with different heuristics)
4. improve_xi_star.py
5. generate_dataset (responsible for creating dataset for ML models)

One should run modules 1 to 5 in order to create the dataset for ML model.
"""
from argparse import ArgumentParser
from configparser import ConfigParser
from importlib import import_module
from pathlib import Path


def main():
    # Load and set configuration
    meta_config, problem_config = ConfigParser(), ConfigParser()
    ROOT = Path(__file__).parent.parent

    # Meta config
    meta_config.read(ROOT / "config" / "meta.ini")
    data_dir = meta_config.get('Directory', 'data')
    problem = meta_config.get('Run', 'problem')
    instance_file = meta_config.get('File', 'instance')
    result_ext_file = meta_config.get('File', 'result_extensive')
    result_xi_file = meta_config.get('File', 'result_xi')

    # Problem config
    problem_config.read(ROOT / "config" / ".".join([problem, "ini"]))
    problem_path = ".".join(["nectar.data_manager", problem])
    get_problem_identifier = getattr(import_module(
        "nectar.utils.combinatorics."+problem
    ), "get_problem_identifier")
    identifier = get_problem_identifier(problem_config)

    # Set path
    data_dir_path = ROOT / data_dir / "_".join([problem, identifier])
    path = {
        "data": data_dir_path,
        "result_xi": data_dir_path / result_xi_file,
        "result_ext": data_dir_path / result_ext_file,
        "instance": data_dir_path / instance_file
    }

    # Specify the module to run
    parser = ArgumentParser()
    parser.add_argument('--run', type=str,
                        help='specify the data_manager module to execute. '
                             'inst: to generate instances '
                             'opt: to generate optimal solution '
                             'repr: to find a representative scenario '
                             'imp: to improve a representative scenario '
                             'dataset : to create dataset for ML'
                             'all: to run all module one after the other ',
                        default='inst')
    args = parser.parse_args()
    if args.run == "inst" or args.run == "all":
        generate_instance = getattr(import_module(".".join([problem_path, "generate_instance"])),
                                    "generate_instance")
        generate_instance(meta_config, problem_config, path)
    if args.run == "opt" or args.run == "all":
        generate_optimal_sol = getattr(import_module(".".join([problem_path, "generate_optimal_sol"])),
                                       "generate_optimal_sol")
        generate_optimal_sol(meta_config, problem_config, path)
    if args.run == "repr" or args.run == "all":
        generate_xi_hat = getattr(import_module(".".join([problem_path, "generate_xi_hat"])),
                                  "generate_xi_hat")
        runs = ConfigParser()
        runs.read(Path(__file__).parents[0] / meta_config['Run']['problem'] / "runs.ini")
        for idx in runs.sections():
            generate_xi_hat(meta_config, runs[idx], path)
    if args.run == "imp" or args.run == "all":
        improve_xi_hat = getattr(import_module(".".join([problem_path, "improve_xi_hat"])),
                                 "improve_xi_hat")
        improve_xi_hat(meta_config, path)
    if args.run == "dataset" or args.run == "all":
        generate_dataset = getattr(import_module(".".join([problem_path, "generate_dataset"])),
                                   "generate_dataset")
        generate_dataset(path)


if __name__ == "__main__":
    main()
