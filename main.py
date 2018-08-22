import argparse
import numpy as np

import test as te
import train as tr

from tqdm import tqdm
from gensim.models import Word2Vec


parser = argparse.ArgumentParser(description='train/test movie review classification model')
parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--n_threads', type=int, help='the number of threads for parsing', default=8)
parser.add_argument('--w2v_model', type=str, help='trained w2v model file', default='ko_w2v.model')
parser.add_argument('--n_dims', type=int, help='embeddings'' dimensions', default=300)
args = parser.parse_args()


class LoadW2VEmbeddings:

    def __init__(self, w2v_model, dims=300):
        self.model = w2v_model

        self.dims = dims

        self.w2v_model = None
        self.embeds = None

        self.load_model()
        self.build_embeds()

    def load_model(self):
        self.w2v_model = Word2Vec.load(self.model)

    def build_embeds(self):
        self.embeds = np.zeros((len(self.w2v_model.wv.vocab), self.dims))

        for i in tqdm(range(len(self.w2v_model.wv.vocab))):
            vec = self.w2v_model.wv[self.w2v_model.wv.index2word[i]]
            if vec is not None:
                self.embeds[i] = vec

    def __len__(self):
        return len(self.w2v_model.wv.vocab)


if __name__ == '__main__':
    # parsed args
    mode = args.mode
    n_dims = args.dims
    trained_w2v_model = args.w2v_model

    w2v = LoadW2VEmbeddings(trained_w2v_model, n_dims)
