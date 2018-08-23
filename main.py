import argparse
import numpy as np
import tensorflow as tf

from .model import charcnn
from .dataloader import Doc2VecEmbeddings


parser = argparse.ArgumentParser(description='train/test movie review classification model')
parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--n_threads', type=int, help='the number of threads', default=8)
parser.add_argument('--model', type=str, help='trained w2v/d2v model file', default='ko_d2v.model')
parser.add_argument('--n_dims', type=int, help='embeddings'' dimensions', default=300)
parser.add_argument('--seed', type=int, help='random seed', default=1337)
args = parser.parse_args()

# parsed args
mode = args.mode
seed = args.seed
vector = args.vec
n_dims = args.dims
vec_model = args.vec_model

np.random.seed(seed)
tf.set_random_seed(seed)


if __name__ == '__main__':
    # Doc2Vec Loader
    vec = Doc2VecEmbeddings(vec_model, n_dims)

    if mode == 'train':
        # GPU configure
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as s:
            model = charcnn.CharCNN(s=s,
                                    n_classes=10,
                                    dims=n_dims)

    elif mode == 'test':
        pass
    else:
        raise ValueError('[-] mode should be train or test')
