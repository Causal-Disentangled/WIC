import argparse
import csv
import numpy as np
import os

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        type=str,
                        default='Alarm',
                        help="Use which dataset.")
    parser.add_argument('--error',
                    type=int,
                    default=3,
                    help="errors added.")                    
    args = parser.parse_args()
    file = r'output_robust_CIR(0)/result_{}_error_{}.csv'.format(args.dataset, args.error)
    print(file)
    with open(file, encoding='utf-8') as f:
        data = np.loadtxt(f, delimiter=',')
    sum0_F1, sum0_SHD = {}, {}
    sum0_F1[200], sum0_F1[500], sum0_F1[1000] = [],[],[]
    sum0_SHD[200], sum0_SHD[500], sum0_SHD[1000] = [],[],[]
    maxF1, minSHD = 0, 10000.0
    for row in data:
        samples = int(data[0])
        F1 = 2*(1-data[2])*data[3]/(1-data[2]+data[3])
        SHD = data[5]
        print(F1)
        print(SHD)
        sum0_F1[samples].append(F1)
        sum0_SHD[samples].append(SHD)

    print('F1:')
    for samples in [200,500,1000]:
        print(samples, np.mean(np.array(sum0_F1[samples])),np.std(np.array(sum0_F1[samples])))
    print('SHD:')
    for samples in [200,500,1000]:
        print(samples, np.mean(np.array(sum0_SHD[samples])),np.std(np.array(sum0_SHD[samples])))




if __name__ == '__main__':
    main()