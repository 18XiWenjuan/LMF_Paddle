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
import pickle
from paddle.io import Dataset


def load_inference_data(data_path, emotion):
    # parse the input args
    class IEMOCAP(Dataset):
        '''
        PyTorch Dataset for IEMOCAP, don't need to change this
        '''
        def __init__(self, audio, visual, text):
            self.audio = audio
            self.visual = visual
            self.text = text

        def __getitem__(self, idx):
            return [self.audio[idx, :], self.visual[idx, :], self.text[idx, :, :]]

        def __len__(self):
            return self.audio.shape[0]

    if sys.version_info.major == 2:
        iemocap_data = pickle.load(open(data_path, 'rb'))
    else:
        iemocap_data = pickle.load(open(data_path, 'rb'), encoding='bytes')

    iemocap_test = iemocap_data[emotion][b'test']

    test_audio, test_visual, test_text \
        = iemocap_test[b'covarep'], iemocap_test[b'facet'], iemocap_test[b'glove']

    test_set = IEMOCAP(test_audio, test_visual, test_text)

    # remove possible NaN values
    test_set.visual[test_set.visual != test_set.visual] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return test_set


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
        self.batch_size = args.batch_size

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
        test_set = load_inference_data(args.data_path, emotion)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, num_workers=0, shuffle=False)
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
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
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

    out_prob = []
    for batch in test_loader:
        x_a = batch[0].astype('float32').cpu().numpy()
        x_v = batch[1].astype('float32').cpu().numpy()
        x_t = batch[2].astype('float32').cpu().numpy()
        # t=
        out_prob.append(inference_engine.run([x_a, x_v, x_t]))

    out_prob = np.concatenate(out_prob)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    return out_prob


if __name__ == "__main__":
    args = get_args()
    out_prob = infer_main(args)
    if not os.path.exists('out_prob'):
        os.mkdir('out_prob')
    np.save('out_prob/{}_prob.npy'.format(args.emotion), out_prob)
