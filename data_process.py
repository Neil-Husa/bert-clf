# -*- coding:utf-8 -*-
# @Author: Neil
# @Time: 2023/2/6 09:59
# @File: data_process.py

import os
# from d2l import torch as d2l
import pandas as pd

# d2l.DATA_HUB['aclImdb'] = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
#                            '01ada507287d82875905620988597833ad4e0903')
# data_dir = d2l.download_extract('aclImdb', 'aclImdb')
data_dir = '/Users/ds/Desktop/HR_2/data/aclImdb'

def read_imdb(data_dir, is_train=True):
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

def create_data(data, is_train=True):
    data_list = []
    for x, y in zip(data[0], data[1]):
        res = (y,x)
        data_list.append(res)
    df = pd.DataFrame(data_list, columns=['lable', 'test'])
    if is_train:
        df.to_csv("data/train.csv")
    else:
        df.to_csv("data/test.csv")

create_data(read_imdb(data_dir))
create_data(read_imdb(data_dir, is_train=False), is_train=False)
