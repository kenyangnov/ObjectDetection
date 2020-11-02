import os, sys, argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt


def draw_pr(pkl_path):
    pr_file = open(pkl_path, 'rb')
    prediction = pk.load(pr_file)
    pr_file.close()
    x = prediction['rec']
    y = prediction['prec']
    plt.figure()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PR cruve')

    x_list = []
    y_list = []
    for i, recall in enumerate(x):
        if recall > 0.05 and y[i] > 0.05:
            x_list.append(recall)
            y_list.append(y[i])
    plt.plot(x_list, y_list, color='red')
    plt.show()
    print('AP：', prediction['ap'])


if __name__ == '__main__':
    classes = [
        'uav',
    ]
    result_path = './pr'
    print("Starting drawing PR curve...")
    for classname in classes:
        pr_file_path = os.path.join(result_path,
                                    '{:s}_pr.pkl'.format(classname))
        draw_pr(pr_file_path)
