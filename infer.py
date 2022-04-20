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

import os
from paddle import inference
import numpy as np
import sys
import os
import paddle
import paddle.nn.functional as F
import numpy as np
import argparse
from paddle.io import DataLoader
from utils import total, load_iemocap
from model import LMF
from sklearn.metrics import f1_score, accuracy_score


class InferenceEngine(object):
    """InferenceEngine
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args
        self.emotion = args.emotion

        # init inference engine
        # save_inference_dir = os.path.abspath(args.save_inference_dir)
        self.predictor, self.config, self.input_tensors, self.output_tensors = self.load_predictor(
            os.path.join(args.save_inference_dir, f"{self.emotion}.pdmodel"),
            os.path.join(args.save_inference_dir, f"{self.emotion}.pdiparams"))

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_tensors = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_tensors = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]

        return predictor, config, input_tensors, output_tensors

    def preprocess(self, args):
        """preprocess
        Preprocess to the input.
        Args:
            data: data.
        Returns: Input data after preprocess.
        """
        emotion = bytes(self.emotion, 'utf-8')
        *_, test_set, input_dims = load_iemocap(args.data_path, emotion)
        test_loader = DataLoader(test_set, batch_size=len(test_set), num_workers=0, shuffle=False)
        return test_loader

    def postprocess(self, x):
        return x

    def run(self, x):
        """run
        Inference process using inference engine.
        Args:
            x: Input data after preprocess.
        Returns: Inference engine output
        """
        self.input_tensors[0].copy_from_cpu(x[0])
        self.input_tensors[1].copy_from_cpu(x[1])
        self.input_tensors[2].copy_from_cpu(x[2])
        self.predictor.run()
        output = self.output_tensors[0].copy_to_cpu()
        return output


def get_args():
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="./data/sample_data.pkl")
    parser.add_argument("--emotion", default='happy')
    parser.add_argument("--save_inference_dir", type=str, default='./check_inference', help="inference model dir")
    parser.add_argument("--use-gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--benchmark", default=False, type=str2bool, help="benchmark")

    args = parser.parse_args()
    return args


def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        label_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="LMF",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # dataset preprocess
    test_loader = inference_engine.preprocess(args)
    if args.benchmark:
        autolog.times.stamp()

    for batch in test_loader:
        x = batch[:-1]
        x_a = x[0].astype('float32')
        x_v = x[1].astype('float32')
        x_t = x[2].astype('float32')
        y = batch[-1].astype('int64')
        # batch_size = x_a.shape[0]
        # y = batch[-1].reshape([batch_size, -1]).astype('int64')
        output_test = inference_engine.run([x_a.cpu().numpy(), x_v.cpu().numpy(), x_t.cpu().numpy()])
    # output_test = output_test.reshape([batch_size, -1])
    # y = y.cpu().numpy().reshape([batch_size, -1])
    y = y.cpu().numpy()
    all_true_label = np.argmax(y, axis=1)
    all_predicted_label = np.argmax(output_test, axis=1)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    all_predicted_label = inference_engine.postprocess(all_predicted_label)

    f1 = f1_score(all_true_label, all_predicted_label, average='weighted')
    acc_score = accuracy_score(all_true_label, all_predicted_label)

    print("F1-score on test set is {}".format(f1))
    print("Accuracy score on test set is {}".format(acc_score))

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    return output_test


if __name__ == "__main__":
    args = get_args()
    output_prob = infer_main(args)
    if not os.path.exists('out_prob'):
        os.mkdir('out_prob')
    np.save('out_prob/{}_prob.npy'.format(args.emotion), output_prob)
