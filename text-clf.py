# -*- coding:utf-8 -*-
# @Author: Neil
# @Time: 2023/2/6 11:36
# @File: text-clf.py
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from trainer import TextClassifizerTrainer
from model import TextClassification
from utils import TextClassifizerDataset
from config import TextClassifizerConfig


#load config from object
config = TextClassifizerConfig()
#tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#load dataset
train_datasets = TextClassifizerDataset(config.train_data,tokenizer, config.max_sequence_length)
eval_datasets = TextClassifizerDataset(config.eval_data, tokenizer, config.max_sequence_length)

#datalodaer
train_dataloader = DataLoader(train_datasets, batch_size=config.batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_datasets, batch_size=config.batch_size, shuffle=True)

#create model
model = TextClassification(config.max_sequence_length, config.num_classes)
#create train
trainer = TextClassifizerTrainer(
    model=model,
    args=None,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    epochs=config.epochs,
    learning_rate=config.learning_rate,
    device=config.device
)

#train model
trainer.train()