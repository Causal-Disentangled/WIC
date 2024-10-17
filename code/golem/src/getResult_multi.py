import argparse
import csv
import os

def getNum(str, sourceStr):

    pos = sourceStr.find(str)
    string = (sourceStr[pos:])
    return string.split(' ')[1][:-1]

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
    return result

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        type=str,
                        default='Alarm',
                        help="Use which dataset.")
    parser.add_argument('--seed',
                        type=int,
                        default=8,
                        help="Use which seed.")
    parser.add_argument('--lambda_1',
                    type=float,
                    default=0.1,
                    help="Coefficient of L1 penalty.")
    parser.add_argument('--lambda_3',
                    type=float,
                    default=1.0,
                    help="Coefficient of CI penalty.")

    args = parser.parse_args()
    file = r'output/{}/1000/non_err0_pequal_varseed_{}/lambda_1_{}lambda_3_{}/training.log'.format(args.dataset, args.seed, args.lambda_1, args.lambda_3)
    result = []
    result.append(args.lambda_1)
    result.append(args.lambda_3)
    result += read_log(file)[-4:]

    flag = os.path.isfile(r'output/{}/1000/result_{}.csv'.format(args.dataset, args.dataset))
    if flag==False:
        f = open('output/{}/1000/result_{}.csv'.format(args.dataset, args.dataset),'w',encoding='utf-8')
        #csv_writer = csv.writer(f)
        #csv_writer.writerow(['lambda_1','lambda_3','fdr','tpr','fpr','shd'])
    
    f = open('output/{}/1000/result_{}.csv'.format(args.dataset, args.dataset),'a+',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(result)



if __name__ == '__main__':
    main()