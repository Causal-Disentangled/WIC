import torch
import numpy as np
import sys
import os
#np.set_printoptions(threshold=np.inf)
import math
from itertools import combinations,permutations
import scipy.stats as st

def getCorr(data, _x, _y):
    x = data[_x]
    y = data[_y]
    cov = np.mean(x * y) - np.mean(x) * np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    if cov>0:
        return np.sqrt(cov**2 / (var_x * var_y))
    else :
        return -np.sqrt(cov**2 / (var_x * var_y))

def indepTest(data, x, y, alpha=0.05):#
    pcc = getCorr(data, x, y)
    zpcc = 0.5*math.log((1+pcc)/(1-pcc))
    A = math.sqrt(data.shape[1] - 3) * math.fabs(zpcc)
    #pc_A = (1 - st.norm.cdf(abs(A))) * 2
    B = st.norm.ppf(1-alpha/2) # Inverse Cumulative Distribution Function of normal Gaussian (parameter : 1-alpha/2)
    #'''
    if A>B:
        return False
    else :
        return True
    #'''
    '''
    if pc_A>0.01:
        return pc_A
    else:
        return 0
    '''

def oldCI_test(data, max_size, x, y, Z):
    num_nodes = len(data)
    corr = np.zeros([num_nodes,num_nodes,max_size])
    vis = np.zeros([num_nodes,num_nodes,max_size],dtype=bool)
    if len(Z) == 0:
        val = getCorr(data, x, y)
        return val      
    
    def getCorr_cond(x, y, z, k): 

        if (vis[x][y][k] == True):
            return corr[x][y][k]
        if (k == len(Z)):
            return getCorr(data, x, y)
        vis[x][y][k] = True
        val_1 = getCorr_cond(x, Z[k], z, k+1)
        val_2 = getCorr_cond(Z[k], y, Z, k+1)
        val = getCorr_cond(x, y, Z, k+1)
        corr[x][y][k] = (val - val_1 * val_2) / (math.sqrt(1 - val_1*val_1) * math.sqrt(1 - val_2*val_2))
        #print('({},{},{}) {:.3f}'.format(i,j,k,corr[x][y][k]))
        return corr[x][y][k]

    val = getCorr_cond(x, y, Z, 0)
    return val

def newCI_test(data, x, y, Z, alpha):
    data_x = np.transpose(data[x]) # num_samples, 转置交换矩阵行列
    data_y = np.transpose(data[y]) # num_samples, 
    data_Z = np.transpose(data[Z,:]) # num_samples * |Z|
    num_samples = data.shape[1]
    
    Z_nodes = len(Z) # length of Z
    if Z_nodes == 0 :
        pcc = getCorr(data, x, y)
    else :
        num_samples = len(data_Z) # number of data samples
        arr_one = (np.ones([num_samples]))
        data_Z = np.insert(data_Z, 0, arr_one, axis = 1) # insert an all-ones column in the left

        wx = np.linalg.lstsq(data_Z, data_x, rcond=None)[0] # wx is the answer of data_Z * X = data_x by using least square method
        wy = np.linalg.lstsq(data_Z, data_y, rcond=None)[0] # wy is the answer of data_Z * X = data_y by using least square method

        rx = data_x - data_Z @ wx # calc residual error of data_x
        ry = data_y - data_Z @ wy # calc residual error of data_y

        pcc = num_samples * (np.transpose(rx) @ ry) - np.sum(rx) * np.sum(ry)
        pcc /= math.sqrt(num_samples * (np.transpose(rx) @ rx) - np.sum(rx) * np.sum(rx))
        pcc /= math.sqrt(num_samples * (np.transpose(ry) @ ry) - np.sum(ry) * np.sum(ry))

    zpcc = 0.5*math.log((1+pcc)/(1-pcc))
    A = math.sqrt(num_samples - Z_nodes - 3) * math.fabs(zpcc)
    #pc_A = (1 - st.norm.cdf(abs(A))) * 2
    B = st.norm.ppf(1-alpha/2) # Inverse Cumulative Distribution Function of normal Gaussian (parameter : 1-alpha/2)
    #'''
    if A>B:
        return False
    else :
        return True
    #'''
    #return pc_A
'''
Input:
    weight : weight matrix
    data : data matrix data.shape: batch_size * num_nodes * 1
    
Output:
    CI_test_val : Float (the CI  test x is independent with y given Z)

Parameter:
    alpha : significance level (default 0.05/0.01)
    if weight[i][j]>alpha:
        regard there is an edge from j to i ???(direction need to be specified)
'''
def prepare_CI_table(data, order, alpha_0, alpha_1):
    num_nodes = data.shape[1]
    data_matrix = np.squeeze(data, axis=-1).transpose()
    CI_table = np.zeros([num_nodes, num_nodes])
    for (x,y) in combinations(range(num_nodes),2):
        CI_table[x][y] = indepTest(data_matrix, x, y, alpha_0)
        CI_table[y][x] = CI_table[x][y]
        if order == 1:
            if CI_table[x][y]==0:
                for z in range(num_nodes):
                    if z==x or z==y:
                        continue
                    if newCI_test(data_matrix, x, y, (z, ), alpha_1)>alpha_1:
                        CI_table[x][y] = newCI_test(data_matrix, x, y, (z, ), alpha_1)
                        CI_table[y][x] = CI_table[x][y]
                        print(CI_table)
                        break

    #np.savetxt('data1.txt', CI_table, fmt='%f', delimiter=',')
    return CI_table

def CI_test_Loss(weight, CI_table):

    num_nodes = weight.shape[0]
    CI_table = torch.from_numpy(CI_table)
    mat = CI_table.mul(weight)
    return torch.sum(mat * mat)


