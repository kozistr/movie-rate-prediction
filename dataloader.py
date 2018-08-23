import numpy as np

from tqdm import tqdm
from gensim.models import Word2Vec, Doc2Vec


class Word2VecEmbeddings:

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


class Doc2VecEmbeddings:

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
