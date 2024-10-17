import argparse
import csv
import numpy as np
import os

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        type=str,
                        default='win95pts',
                        help="Use which dataset.")
    args = parser.parse_args()
    file = r'output_CIR(1)/result_{}(0-order).csv'.format(args.dataset)
    print(file)
    with open(file, encoding='utf-8') as f:
        data = np.loadtxt(f, delimiter=',')
    sum0_F1, sum0_SHD = {}, {}
    sum1_F1, sum1_SHD = {}, {}
    sum0_F1[200], sum0_F1[500], sum0_F1[1000] = 0,0,0
    sum0_SHD[200], sum0_SHD[500], sum0_SHD[1000] = 0,0,0
    sum1_F1[200], sum1_F1[500], sum1_F1[1000] = 0,0,0
    sum1_SHD[200], sum1_SHD[500], sum1_SHD[1000] = 0,0,0
    maxF1, minSHD = 0, 10000.0
    for row in data:
        print(data)
        samples = int(data[0])
        F1 = 2*(1-data[2])*data[3]/(1-data[2]+data[3])
        print(F1)
        SHD = data[5]
        print(SHD)
        if data[1] == 0.0:
            sum0_F1[samples] += F1
            sum0_SHD[samples] += SHD
            maxF1 = 0
            minSHD = 10000.0
        else:
            if F1 > maxF1:
                maxF1 = F1
            if SHD < minSHD:
                minSHD = SHD
            if data[1] == 1.0:
                sum1_F1[samples] += maxF1
                sum1_SHD[samples] += minSHD
                maxF1 = 0
                minSHD = 10000.0
    print('F1:')
    for samples in [200,500,1000]:
        print(samples, sum0_F1[samples]/6)
    print('SHD:')
    for samples in [200,500,1000]:
        print(samples, sum0_SHD[samples]/6)

    print('------')

    print('F1:')
    for samples in [200,500,1000]:
        print(samples, sum1_F1[samples]/6)
    print('SHD:')
    for samples in [200,500,1000]:
        print(samples, sum1_SHD[samples]/6)
                
            



if __name__ == '__main__':
    main()