import os, sys, argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt

def draw_pr(pkl_path):
    pr_file = open(pkl_path,'rb')
    prediction = pk.load(pr_file)
    pr_file.close()
    x=prediction['rec']
    y=prediction['prec']
    plt.figure()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PR cruve')
    print('recall')
    print(x)
    print('precision')
    print(y)
    '''
    for i, data in enumerate(x):
        print("recall:",data,"precision:", y[i])
    np.savetxt('recall.txt',x)
    np.savetxt('precision.txt',y)
    x_list = []
    y_list = []
    for i, recall in enumerate(x):
        if recall>0.05 and y[i] > 0.05:
            x_list.append(recall)
            y_list.append(y[i])
    plt.plot(x_list,y_list,color = 'red')
    '''

    plt.plot(x,y,color = 'red')
    plt.show()
    print('AP：',prediction['ap'])

if __name__ == '__main__':
    result_path = './result'
    pr_file_path = os.path.join(result_path,'uav_pr.pkl')

    print("开始画图")
    draw_pr(pr_file_path)

