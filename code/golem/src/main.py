"""Each run creates a directory based on current datetime to save:
- log file of training process
- experiment configurations
- observational data and ground truth
- final estimated solution
- visualization of final estimated solution
- intermediate training outputs (e.g., estimated solution).
"""

import logging

import numpy as np

from data_loader import SyntheticDataset
from golem import golem
from utils.config import save_yaml_config, get_args
from utils.dir import create_dir, get_datetime_str
from utils.logger import LogHelper
from utils.train import postprocess, checkpoint_after_training
from utils.utils import set_seed, get_init_path
from utils.CI import *
from cdt.metrics import SID
import sys


def main():
    # Get arguments parsed
    args = get_args()

    # Setup for logging
    output_dir = 'output/{}/{}/non_err0_pequal_varseed_{}/lambda_1_{}lambda_3_{}'.format(args.dataset, args.n, args.seed, args.lambda_1, args.lambda_3)
    create_dir(output_dir)    # Create directory to save log files and outputs
    LogHelper.setup(log_path='{}/training.log'.format(output_dir), level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger.")

    # Save configs
    save_yaml_config(vars(args), path='{}/config.yaml'.format(output_dir))

    # Reproducibility
    set_seed(args.seed)
    # Load dataset
    dataset = SyntheticDataset(args.n, args.d, args.dataset, args.graph_type, args.degree,
                               args.noise_type, args.B_scale, args.seed)
    _logger.info("Finished loading the dataset.")
    args.n = dataset.X.shape[0]
    args.d = dataset.X.shape[1]

    # Load B_init for initialization
    if args.init:
        if args.init_path is None:
            args.init_path = get_init_path('output/')

        B_init = np.load('{}'.format(args.init_path))
        _logger.info("Finished loading B_init from {}.".format(args.init_path))
    else:
        B_init = None
    print(B_init)
    

    # Calculate the CI table of dataset.X
    dataset.X -= np.mean(dataset.X, axis=0)
    # np.savetxt('data.csv', dataset.X, delimiter=',')
    CI_table = prepare_CI_table(dataset.X, alpha = 0.01)
    #print(CI_table)

    row, column = np.where(dataset.B!=0) 
    sz = len(row)
    cnt = 0
    while cnt < 0:
        idx= torch.empty([1], dtype=torch.long).random_(sz)
        x0 = row[idx] 
        y0 = column[idx]
        while not (CI_table[x0][y0]==0):
            idx= torch.empty([1], dtype=torch.long).random_(sz)
            x0 = row[idx] 
            y0 = column[idx]
        CI_table[x0][y0] = 1
        cnt += 1

    # Check the correctness of CI_table
    ground_truth = np.ones([args.d, args.d]) - np.eye(args.d) - (dataset.B!=0)
    ground_truth = ground_truth * ground_truth.transpose()
    ground_truth = np.ones([args.d, args.d]) - np.eye(args.d) - ground_truth
    print('Type II error edges: ',np.sum(ground_truth * CI_table))
    
    # cut off some true edges
    # skeleton_CI = np.ones([args.d, args.d]) - np.eye(args.d) - CI_table
    # row, column = np.where(skeleton_CI!=0)
    # print(row)
    # print(column)
    # sys.exit(0)

    # GOLEM
    B_est = golem(dataset.X, CI_table, args.lambda_1, args.lambda_2, args.lambda_3, args.equal_variances, args.num_iter,
                  args.learning_rate, args.seed, args.checkpoint_iter, output_dir, B_init)
    _logger.info("Finished training the model.")

    # Post-process estimated solution
    B_processed = postprocess(B_est, args.graph_thres)
    _logger.info("Finished post-processing the estimated graph.")

    # Checkpoint
    checkpoint_after_training(output_dir, dataset.X, dataset.B, B_init,
                              B_est, B_processed, _logger.info)
    #_logger.info("SID :{}".format(SID(dataset.B, B_processed!=0)))

if __name__ == '__main__':
    main()
