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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import os
import sys
import numpy as np

paddle.set_device("gpu")

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import paddle.vision
from model import LMF
from utils import load_iemocap

def get_args():
    import argparse
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--emotion', dest='emotion', type=str, default='happy')
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=100)
    OPTIONS.add_argument('--output_dim', dest='output_dim', type=int, default=2)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--data_path', dest='data_path', type=str, default='data/sample_data.pkl')
    OPTIONS.add_argument('--check_path', dest='check_path', type=str, default='./check/')
    OPTIONS.add_argument('--save_inference_dir', dest='save_inference_dir', type=str, default='./check_inference/')
    OPTIONS.add_argument('--pretrained', dest='pretrained', type=str, default='null')
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

    args = OPTIONS.parse_args()
    return args


def export(args):
    data_path = args.data_path
    check_path = args.check_path
    emotion = args.emotion
    model_name = os.path.join(check_path, emotion+'.pkl')
    b_emotion = bytes(emotion, 'utf-8')
    output_dim = args.output_dim
    ahid = args.audio_hidden
    vhid = args.video_hidden
    thid = args.text_hidden
    thid_2 = thid // 2
    adr = args.audio_dropout
    vdr = args.video_dropout
    tdr = args.text_dropout
    r = args.rank

    train_set, valid_set, test_set, input_dims = load_iemocap(data_path, b_emotion)
    model = LMF(input_dims, (ahid, vhid, thid), thid_2, (adr, vdr, tdr, 0.5), output_dim, r)
    checkpoint = paddle.load(model_name)
    model.set_dict(checkpoint)
    model.eval()

    model = paddle.jit.to_static(
        model,
        input_spec=[
            InputSpec(shape=[None, 74], dtype='float32'),
            InputSpec(shape=[None, 35], dtype='float32'),
            InputSpec(shape=[None, 20, 300], dtype='float32'),
        ])

    if not os.path.exists(args.save_inference_dir):
        os.mkdir(args.save_inference_dir)

    paddle.jit.save(model, os.path.join(args.save_inference_dir, f'{emotion}'))
    print(f"inference model has been saved into {args.save_inference_dir}")


if __name__ == "__main__":
    args = get_args()
    export(args)