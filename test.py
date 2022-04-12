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
from utils import total, load_iemocap
from sklearn.metrics import accuracy_score, f1_score
import os
import argparse
import numpy as np

import paddle
from paddle.io import DataLoader


def display(f1_score, accuracy_score):
    print("F1-score on test set is {}".format(f1_score))
    print("Accuracy score on test set is {}".format(accuracy_score))


def main(options):
    # parse the input args
    data_path = options['data_path']
    check_path = options['check_path']
    emotion = options['emotion']
    output_dim = options['output_dim']
    ahid = options['audio_hidden']
    vhid = options['video_hidden']
    thid = options['text_hidden']
    thid_2 = thid // 2
    adr = options['audio_dropout']
    vdr = options['video_dropout']
    tdr = options['text_dropout']
    r = options['rank']

    print('Start testing {}......'.format(emotion))

    emotion = bytes(emotion, 'utf-8')
    # 数据读入
    train_set, valid_set, test_set, input_dims = load_iemocap(data_path, emotion)
    model = LMF(input_dims, (ahid, vhid, thid), thid_2, (adr, vdr, tdr, 0.5), output_dim, r)
    if options['cuda']:
        paddle.device.set_device('gpu:0')
    print("Model initialized")
    # 数据加载
    test_iterator = DataLoader(test_set, batch_size=len(test_set), num_workers=4, shuffle=True)
    # 模型加载
    checkpoint = paddle.load(check_path)
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
    OPTIONS.add_argument('--signiture', dest='signiture', type=str, default='iemocap')
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--data_path', dest='data_path',
                         type=str, default='data/')
    OPTIONS.add_argument('--check_path', dest='check_path',
                         type=str, default='check/happy.pkl')
    OPTIONS.add_argument('--output_path', dest='output_path',
                         type=str, default='results')
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
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
