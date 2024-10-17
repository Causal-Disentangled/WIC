import argparse
import csv
import os

def getNum(str, sourceStr):

    pos = sourceStr.find(str)
    string = (sourceStr[pos:])
    str_cut = string.split(' ')
    for s in str_cut[1:]:
        if s!='':
            # print(str, s)
            return s

def read_log(file):
    result = []
    with open(file) as f:
        while True:
            lines = f.readline()
            if not lines:
                break
            if 'fdr' in lines:
                result.append(getNum('fdr',lines))
                result.append(getNum('tpr',lines))
                result.append(getNum('fpr',lines))
                result.append(getNum('shd',lines))
    # print(result[:-4])
    return result[-4:]

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        type=str,
                        help="Use which dataset.")
    parser.add_argument('--samples',
                    type=int,
                    default=0,
                    help="number of samples.")
    parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help="random seed.")
    parser.add_argument('--lambda_CI',
                    type=float,
                    default=0.0,
                    help="Coefficient of CI penalty.")

    args = parser.parse_args()
    file = r'output_CIC(0)/{}/seed:{}.tau:0.0/samples:{} lambda_CI:{}/log.txt'.format(args.dataset, args.seed, args.samples, args.lambda_CI)
    result = []
    result.append(args.samples)
    result.append(args.lambda_CI)
    result += read_log(file)

    flag = os.path.isfile(r'output_CIC(0)/result_{}_error_0.csv'.format(args.dataset))
    if flag==False:
        f = open('output_CIC(0)/result_{}_error_0.csv'.format(args.dataset),'w',encoding='utf-8')
        csv_writer = csv.writer(f)
        # csv_writer.writerow(['samples','lambda_CI','fdr','tpr','fpr','shd'])
    
    f = open('output_CIC(0)/result_{}_error_0.csv'.format(args.dataset),'a+',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(result)



if __name__ == '__main__':
    main()