import argparse
import sys

import yaml


def load_yaml_config(path):
    """Load the config file in yaml format.

    Args:
        path (str): Path to load the config file.

    Returns:
        dict: config.
    """
    with open(path, 'r') as infile:
        return yaml.safe_load(infile)


def save_yaml_config(config, path):
    """Load the config file in yaml format.

    Args:
        config (dict object): Config.
        path (str): Path to save the config.
    """
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def get_args():
    """Add arguments for parser.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    add_dataset_args(parser)
    add_model_args(parser)
    add_training_args(parser)
    add_other_args(parser)

    return parser.parse_args(args=sys.argv[1:])


def add_dataset_args(parser):
    """Add dataset arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    parser.add_argument('--n',
                    type=int,
                    default=1000,
                    help="Number of samples.")

    parser.add_argument('--d',
                        type=int,
                        default=20,
                        help="Number of nodes.")

    parser.add_argument('--dataset',
                        type=str,
                        default='hailfinder',
                        help="Use which dataset.")

    parser.add_argument('--graph_type',
                        type=str,
                        default='ER',
                        help="Type of graph ('ER' or 'SF').")

    parser.add_argument('--degree',
                        type=int,
                        default=4,
                        help="Degree of graph.")

    parser.add_argument('--noise_type',
                        type=str,
                        default='uniform',
                        help="Type of noise ['uniform', 'gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'].")

    parser.add_argument('--B_scale',
                        type=float,
                        default=1.0,
                        help="Scaling factor for range of B.")


def add_model_args(parser):
    """Add model arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    parser.add_argument('--init',
                        dest='init',
                        default=True,
                        action='store_true',
                        help="Whether to initialize the optimization with a given weighted matrix.")

    parser.add_argument('--init_path',
                        type=str,
                        default='output/hailfinder/1000/err0_pequal_varseed_9/lambda_1_0.05lambda_3_0.5/B_est.npy',
                        help="Path of weighted matrix for initialization. Set to None to disable.")
    # 'output/Alarm/1000/equal_varseed_1/best_6_lambda_1_0.05lambda_3_0.5/B_est.npy'  win95pts  hailfinder

    parser.add_argument('--seed',
                        type=int,
                        default=9,
                        help="Random seed.")

    parser.add_argument('--lambda_1',
                        type=float,
                        default=0.1,
                        help="Coefficient of L1 penalty.")

    parser.add_argument('--lambda_2',
                        type=float,
                        default=5.0,
                        help="Coefficient of DAG penalty.")
    
    parser.add_argument('--lambda_3',
                        type=float,
                        default=1.0,
                        help="Coefficient of CI penalty.")

    parser.add_argument('--equal_variances',
                        dest='equal_variances',
                        default=False,
                        action='store_true',
                        help="Assume equal noise variances for likelibood objective.")#False

    parser.add_argument('--non_equal_variances',
                        dest='equal_variances',
                        default=True,
                        action='store_false',
                        help="Assume non-equal noise variances for likelibood objective.")#True


def add_training_args(parser):
    """Add training arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help="Learning rate of Adam optimizer.")

    parser.add_argument('--num_iter',
                        type=int,
                        default=40000,
                        help="Number of iterations for training.")

    parser.add_argument('--checkpoint_iter',
                        type=int,
                        default=5000,
                        help="Number of iterations between each checkpoint. Set to None to disable.")


def add_other_args(parser):
    """Add other arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """

    parser.add_argument('--graph_thres',
                        type=float,
                        default=0.3,
                        help="Threshold for weighted matrix.")
