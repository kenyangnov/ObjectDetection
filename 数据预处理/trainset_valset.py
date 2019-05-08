#!/usr/bin/env python3
# coding=UTF-8
import pandas as pd
import random
import os
import shutil

if not os.path.exists('trainset/'):
    os.mkdir('trainset/')

if not os.path.exists('valset/'):
    os.mkdir('valset/')

val_rate = 0.15

img_path = 'train/'
img_list = os.listdir(img_path)
train = pd.read_csv('train_label_fix.csv')
# print(train.head(5))
# print(img_list)

random.shuffle(img_list)

total_num = len(img_list)
val_num = int(total_num*val_rate)
train_num = total_num-val_num

for i in range(train_num):
    img_name = img_list[i]
    shutil.copy('train/' + img_name, 'trainset/' + img_name)
for j in range(val_num):
    img_name = img_list[j+train_num]
    shutil.copy('train/' + img_name, 'valset/' + img_name)

print("Finished!")