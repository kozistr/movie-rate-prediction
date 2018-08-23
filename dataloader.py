import gc
import psutil
import numpy as np

from tqdm import tqdm
from konlpy.tag import Mecab
from soynlp.normalizer import *
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

    def __init__(self, d2v_model, dims=300):
        self.model = d2v_model

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


class DataLoader:

    def __init__(self, file, save_to_file=False, n_threads=8, mem_limit=256):
        self.file = file
        assert self.file in '.csv'

        self.data = []

        self.sentences = []
        self.labels = []

        self.save_to_file = save_to_file
        self.n_threads = n_threads
        self.mem_limit = mem_limit

        self.remove_dirty()
        self.word_tokenize()

    def remove_dirty(self):
        with open(self.file, 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines()[1:]):
                d = line.split(',')
                try:
                    # remove dirty stuffs
                    self.data.append({'rate': d[0], 'comment': ','.join(d[1:]).replace('\x00', '').replace('\n', '').
                                     replace('<span class=""ico_penel""></span>', '').strip('"').strip()})
                except Exception as e:
                    print(e, line)
                del d

    def word_tokenize(self):
        # Mecab Pos Tagger
        mecab = Mecab()

        def emo(x: str) -> str:
            return emoticon_normalize(x, n_repeats=3)

        def rep(x: str) -> str:
            return repeat_normalize(x, n_repeats=2)

        def normalize(x: str) -> str:
            return rep(emo(x))

        n_data = len(self.data)
        for idx, d in enumerate(self.data):
            pos = list(map(lambda x: '/'.join(x), mecab.pos(normalize(d['comment']))))

            # append sentence & rate
            self.sentences.append(pos)
            self.labels.append(d['rate'])

            if idx > 0 and idx % (n_data // (100 * self.n_threads)) == 0:
                print("[*] %d/%d" % (idx, n_data), pos)
                gc.collect()

                remain_ram = psutil.virtual_memory().available / (2 ** 20)
                if remain_ram < self.mem_limit:
                    import sys
                    print("[-] not enough memory < 256MB, ", remain_ram)
                    sys.exit(-1)
            del pos
