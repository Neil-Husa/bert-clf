# -*- coding:utf-8 -*-
# @Author: Neil
# @Time: 2023/2/6 11:41
# @File: utils.py
import collections
import os
import re
from typing import Text, Optional, Dict, Set
import pandas as pd
import torch
from torch.utils.data import Dataset

class TextClassifizerDataset(Dataset):

    def __init__(self, filepath, tokenizer, max_length):
        self.max_length = max_length
        self.dataframe = pd.read_csv(filepath)
        self.text_dir = filepath
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        labels = self.dataframe.iloc[idx, 0]
        text = self.dataframe.iloc[idx, 1]
        token = self.tokenizer(text, return_tensors='pt',
                               padding='max_length',
                               max_length=self.max_length,
                               truncation=True)
        return {'labels': torch.tensor(labels, dtype=torch.long),
                'token':token}


    def num_classes(self):
        return len(set(self.dataframe['lable']))