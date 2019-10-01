# coding=utf-8
import os, sys, argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

def parse_args():
    """
    解析输入参数
    """
    parser = argparse.ArgumentParser(description='Drawing Curve')
    parser.add_argument('net_version',type = str)
    args = parser.parse_args()
    return args



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
    
    '''
    x_list = []
    y_list = []
    for i, recall in enumerate(x):
        if recall>0.05 and y[i] > 0.05:
            x_list.append(recall)
            y_list.append(y[i])
    plt.plot(x_list,y_list,color = 'red')
    '''
    #折线平滑报错
    #x_new = np.linspace(min(x_list),max(x_list),100000)
    #y_new = spline(x_list, y_list, x_new)
    #plt.plot(x_new, y_new)

    #plt.ylim(0,1)
    #plt.xlim(0,1)
    plt.plot(x,y,color = 'red')
    #plt.plot(x2,y2,color = 'blue')
    plt.show()

    print('AP：',prediction['ap'])

if __name__ == '__main__':
    args = parse_args()
    
    net_version = args.net_version
    result_root = '/home/wl/Desktop/EXTD/result'
    pr_file_path = os.path.join(result_root, net_version, 'uav_pr.pkl')

    print("开始画图")
    draw_pr(pr_file_path)

