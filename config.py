# -*- coding:utf-8 -*-
# @Author: Neil
# @Time: 2023/2/1 10:42
# @File: config.py

from typing import Text, Dict, Optional
import torch


class TextClassifizerConfig():

    def __init__(self,
                 num_classes: int = 3,
                 batch_size: int = 4,
                 learning_rate=20,
                 epochs: int = 20,
                 max_sequence_length: int = 100,
                 train_data: Text = "data/text-classifizer/train.csv",
                 eval_data: Text = "data/text-classifizer/train.csv"):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_sequence_length = max_sequence_length
        self.eval_data = eval_data
