import argparse
import numpy as np
import tensorflow as tf

from model import charcnn
from dataloader import Doc2VecEmbeddings, DataLoader


parser = argparse.ArgumentParser(description='train/test movie review classification model')
parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--dataset', type=str, help='DataSet path', default='./data.csv')
parser.add_argument('--n_threads', type=int, help='the number of threads', default=8)
parser.add_argument('--model', type=str, help='trained w2v/d2v model file', default='ko_d2v.model')
parser.add_argument('--n_dims', type=int, help='embeddings'' dimensions', default=300)
parser.add_argument('--seed', type=int, help='random seed', default=1337)
parser.add_argument('--save_to_file', type=bool, help='save DataSet into .csv file', default=True)
parser.add_argument('--save_file', type=str, help='DataSet file name', default='tagged_data.csv')
args = parser.parse_args()

# parsed args
mode = args.mode
seed = args.seed
n_dims = args.dims
dataset = args.dataset
vec_model = args.vec_model
n_threads = args.n_threads

save_to_file = args.save_to_file
save_file = args.save_file

np.random.seed(seed)
tf.set_random_seed(seed)


if __name__ == '__main__':
    # DataSet Loader
    ds = DataLoader(dataset,
                    save_to_file=save_to_file,
                    save_file=save_file,
                    n_threads=n_threads)
    if save_file:
        import sys
        sys.exit(0)

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
        print('[-] mode should be train or test')
