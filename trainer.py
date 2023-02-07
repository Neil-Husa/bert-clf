# -*- coding:utf-8 -*-
# @Author: Neil
# @Time: 2023/2/1 11:33
# @File: trainer.py
import time

import torch
from typing import Union, Optional, Text, Tuple
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):

    def name(self) -> Text:
        raise NotImplementedError

    def save(self,
             model: torch.nn.Module = None,
             optimizer=None,
             epoch: Optional[int] = None):
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch}
        path_checkpoint = f"models/{self.name()}-checkpoint_{epoch}_epoch.pkl"
        torch.save(checkpoint, path_checkpoint)


class TextClassifizerTrainer(Trainer):

    def name(self) -> Text:
        return "text_classifizer"

    def __init__(self, model: torch.nn.Module = None,
                 args: Optional[Tuple] = None,
                 train_dataloader: DataLoader = None,
                 eval_dataloader: DataLoader = None,
                 epochs: Optional[int] = 30,
                 learning_rate: Optional[float] = 1e-5,
                 device: Optional[Text] = 'cpu'):

        self.writer = SummaryWriter(
            f'logs/text-classifier-B-{train_dataloader.batch_size}-E{epochs}-L{learning_rate}-{time.time()}'
        )
        self.writer.flush()
        if model is None:
            raise RuntimeError("`Trainer` requires a `model` ")
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device

    def train(self):
        loss_fn = torch.nn.CrossEntropyLoss()  # 多分类更换函数
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=0.1)

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.train_loop(epoch, loss_fn, optimizer)
            accu_val, loss = self.eval_loop(epoch, loss_fn)
            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                  'valid accuracy {:8.3f}'
                  'valid loss {:8.3f}'
                  .format(epoch, time.time() - epoch_start_time,
                          accu_val, loss))
            print('-' * 59)
            self.save(model=self.model, optimizer=optimizer, epoch=epoch)

    def train_loop(self, epoch, loss_fn, optimizer):
        self.model.train()
        total_acc, total_count = 0, 0
        for batch, data in enumerate(self.train_dataloader):
            y = data['labels'].to(self.device)
            token = data['token']
            input_ids = token['input_ids'].squeeze(1).to(self.device)
            attention_mask = token['attention_mask'].squeeze(1).to(self.device)
            token_type_ids = token['token_type_ids'].squeeze(1).to(self.device)

            # compute prediction and loss
            pred = self.model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(pred, y)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_acc = (pred.argmax(1) == y).sum().item()
            current_count = y.size(0)
            loss, current = loss.item(), batch * len(token)

            total_acc += current_acc
            total_count += current_count

            # log the runing loss
            self.writer.add_scalar(
                'training loss',
                loss,
                (epoch - 1) * len(self.train_dataloader) + batch
            )
            # log a matplotlig figure showing the model's prediction on a random mini-batch
            self.writer.add_scalar('training acc',
                                   current_acc / current_count,
                                   (epoch - 1) * len(self.train_dataloader) + batch
                                   )
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'
                  '| loss {:8.3f}'
                  .format(epoch, batch, len(self.train_dataloader),
                          current_acc / current_count, loss))

    def eval_loop(self, epoch, loss_fn):
        self.model.eval()
        total_acc, total_count = 0, 0
        loss = 0

        with torch.no_grad():
            for batch, data in enumerate(self.eval_dataloader):
                y = data['labels'].to(self.device)
                token = data["token"]
                input_ids = token["input_ids"].squeeze(1).to(self.device)
                attention_mask = token["attention_mask"].squeeze(1).to(self.device)
                token_type_ids = token["token_type_ids"].squeeze(1).to(self.device)

                # Compute prediction and loss
                pred = self.model(input_ids, attention_mask, token_type_ids)
                loss = loss_fn(pred, y)
                loss, current = loss.item(), batch * len(token)
                current_acc = (pred.argmax(1) == y).sum().item()
                current_count = y.size(0)

                total_acc += current_acc
                total_count += current_count

                # ...log the running loss
                self.writer.add_scalar('eval loss',
                                       loss,
                                       (epoch - 1) * len(self.eval_dataloader) + batch)

                self.writer.add_scalar('eval acc',
                                       current_acc / current_count,
                                       (epoch - 1) * len(self.eval_dataloader) + batch)

        return total_acc / total_count, loss
