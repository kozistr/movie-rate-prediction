import argparse
import numpy as np

from tqdm import tqdm
from gensim.models import Word2Vec, Doc2Vec

from .model import charcnn


parser = argparse.ArgumentParser(description='train/test movie review classification model')
parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--n_threads', type=int, help='the number of threads for parsing', default=8)
parser.add_argument('--model', type=str, help='trained w2v/d2v model file', default='ko_w2v.model')
parser.add_argument('--n_dims', type=int, help='embeddings'' dimensions', default=300)
parser.add_argument('--vector', type=str, help='word2vec or doc2vec', default='d2v')
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


class LoadD2VEmbeddings:

    def __init__(self, w2v_model, dims=300):
        self.model = w2v_model

        self.dims = dims

        self.d2v_model = None
        self.embeds = None

        self.load_model()

    def load_model(self):
        self.d2v_model = Doc2Vec.load(self.model)

    def sentence_to_vector(self, input_sentence: str) -> np.array:
        return self.d2v_model.infer_vector(input_sentence)

    def __len__(self):
        return len(self.d2v_model.wv.vocab)


if __name__ == '__main__':
    # parsed args
    mode = args.mode
    vector = args.vec
    n_dims = args.dims
    vec_model = args.vec_model

    if vector == 'w2v':
        vec = LoadW2VEmbeddings(vec_model, n_dims)
    elif vector == 'd2v':
        vec = LoadD2VEmbeddings(vec_model, n_dims)
    else:
        raise ValueError("[-] vector must be w2v or d2v")

    if mode == 'train':
        model = charcnn.CharCNN(dims=n_dims,
                                use_w2v=True,
                                w2v_model=vec)

    elif mode == 'test':
        pass
    else:
        raise ValueError('[-] mode should be train or test')
