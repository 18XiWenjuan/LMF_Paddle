# MIT License
#
# Copyright (c) 2022 18XiWenjuan
#
# Permission is hereby granted, free of charge, to any person obtaining a cop
# y
# of this software and associated documentation files (the "Software"), to de
# al
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in al
# l
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIN
# D, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHA
# NTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO E
# VENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGE
# S OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM,
# OUT OF OR IN CONNECTION W

from __future__ import print_function
from model import LMF
from utils import load_iemocap
from sklearn.metrics import accuracy_score, f1_score
import os
import argparse
import numpy as np
import random
import paddle
import paddle.nn as nn
from paddle.io import DataLoader


def display(f1_score, accuracy_score):
    print("F1-score on test set is {}".format(f1_score))
    print("Accuracy score on test set is {}".format(accuracy_score))

def main(options):
    paddle.seed(options['seed'])
    np.random.seed(options['seed'])
    random.seed(options['seed'])
    # parse the input args
    epochs = options['epochs']
    data_path = options['data_path']
    model_path = options['model_path']
    patience = options['patience']
    emotion = options['emotion']
    output_dim = options['output_dim']
    ahid = options['audio_hidden']
    vhid = options['video_hidden']
    thid = options['text_hidden']
    thid_2 = thid // 2
    adr = options['audio_dropout']
    vdr = options['video_dropout']
    tdr = options['text_dropout']
    factor_lr = options['factor_learning_rate']
    lr = options['learning_rate']
    r = options['rank']
    batch_sz = options['batch_size']
    decay = options['weight_decay']
    model_path = os.path.join(model_path, "{}.pkl".format(emotion))

    emotion = bytes(emotion, 'utf-8')
    # 数据读入
    train_set, valid_set, test_set, input_dims = load_iemocap(data_path, emotion)
    model = LMF(input_dims, (ahid, vhid, thid), thid_2, (adr, vdr, tdr, 0.5), output_dim, r)
    if options['cuda']:
        paddle.device.set_device('gpu:0')
    print("Model initialized")
    criterion = nn.CrossEntropyLoss(reduction='sum')
    factors = list(model.parameters())[:3]
    other = list(model.parameters())[3:]
    optimizer = paddle.optimizer.Adam(
        parameters=[{"params": factors, "lr": factor_lr}, {"params": other, "lr": lr}], weight_decay=decay)

    # setup training
    complete = True
    min_valid_loss = float('Inf')
    curr_patience = patience
    # 数据加载
    train_iterator = DataLoader(train_set, batch_size=batch_sz, num_workers=4, shuffle=True)
    valid_iterator = DataLoader(valid_set, batch_size=len(valid_set), num_workers=4, shuffle=True)
    test_iterator = DataLoader(test_set, batch_size=len(test_set), num_workers=4, shuffle=True)

    for e in range(epochs):
        model.train()
        avg_train_loss = 0.0
        for batch in train_iterator:
            optimizer.clear_grad()

            x = batch[:-1]
            x_a = paddle.to_tensor(x[0], dtype='float32')
            x_v = paddle.to_tensor(x[1], dtype='float32')
            x_t = paddle.to_tensor(x[2], dtype='float32')
            y = paddle.to_tensor(batch[-1], dtype='int64')
            output = model(x_a, x_v, x_t)
            # loss计算
            loss = criterion(output, paddle.argmax(y, 1))
            loss.backward()
            avg_loss = loss.item()
            avg_train_loss += avg_loss / len(train_set)
            optimizer.step()

        print("Epoch {} complete! Average Training loss: {}".format(e, avg_train_loss))

        # Terminate the training process if run into NaN
        if np.isnan(avg_train_loss):
            print("Training got into NaN values...\n\n")
            complete = False
            break

        model.eval()
        for batch in valid_iterator:
            x = batch[:-1]
            x_a = paddle.to_tensor(x[0], dtype='float32')
            x_v = paddle.to_tensor(x[1], dtype='float32')
            x_t = paddle.to_tensor(x[2], dtype='float32')
            y = paddle.to_tensor(batch[-1], dtype='int64')
            output = model(x_a, x_v, x_t)
            valid_loss = criterion(output, paddle.argmax(y, 1))
            avg_valid_loss = valid_loss.item()

        if np.isnan(avg_valid_loss):
            print("Training got into NaN values...\n\n")
            complete = False
            break

        avg_valid_loss = avg_valid_loss / len(valid_set)
        print("Validation loss is: {}".format(avg_valid_loss))

        if (avg_valid_loss < min_valid_loss):
            curr_patience = patience
            min_valid_loss = avg_valid_loss
            # 保存模型
            paddle.save(model.state_dict(), os.path.join(model_path))
            print("Found new check model, saving to disk...")
        else:
            curr_patience -= 1

        if curr_patience <= 0:
            break

    if complete:
        # 加载模型
        checkpoint = paddle.load(os.path.join(model_path))
        model.set_state_dict(checkpoint)
        model.eval()
        for batch in test_iterator:
            x = batch[:-1]
            x_a = paddle.to_tensor(x[0], dtype='float32')
            x_v = paddle.to_tensor(x[1], dtype='float32')
            x_t = paddle.to_tensor(x[2], dtype='float32')
            y = paddle.to_tensor(batch[-1], dtype='int64')
            output_test = model(x_a, x_v, x_t)

        # these are the needed metrics
        all_true_label = np.argmax(y, axis=1)
        all_predicted_label = np.argmax(output_test, axis=1)

        f1 = f1_score(all_true_label, all_predicted_label, average='weighted')
        acc_score = accuracy_score(all_true_label, all_predicted_label)

        display(f1, acc_score)


if __name__ == "__main__":
    # 参数定义
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--emotion', dest='emotion', type=str, default='happy')
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=100)
    OPTIONS.add_argument('--output_dim', dest='output_dim', type=int, default=2)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--data_path', dest='data_path',
                         type=str, default='data/iemocap.pkl')
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='check')
    OPTIONS.add_argument('--audio_hidden', dest='audio_hidden', type=int, default=4)
    OPTIONS.add_argument('--video_hidden', dest='video_hidden', type=int, default=16)
    OPTIONS.add_argument('--text_hidden', dest='text_hidden', type=int, default=128)
    OPTIONS.add_argument('--audio_dropout', dest='audio_dropout', type=float, default=0.3)
    OPTIONS.add_argument('--video_dropout', dest='video_dropout', type=float, default=0.1)
    OPTIONS.add_argument('--text_dropout', dest='text_dropout', type=float, default=0.5)
    OPTIONS.add_argument('--factor_learning_rate', dest='factor_learning_rate', type=float, default=0.001)
    OPTIONS.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
    OPTIONS.add_argument('--rank', dest='rank', type=int, default=1)
    OPTIONS.add_argument('--batch_size', dest='batch_size', type=int, default=128)
    OPTIONS.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.002)
    OPTIONS.add_argument('--seed', dest='seed', type=int, default=1)
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
