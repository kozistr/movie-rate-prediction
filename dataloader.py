import gc
import numpy as np

from tqdm import tqdm
from konlpy.tag import Mecab
from soynlp.normalizer import *
from multiprocessing import Pool
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

    def __init__(self, file, save_to_file=False, save_file=None, n_threads=8, mem_limit=256):
        self.file = file
        assert self.file in '.csv'

        self.data = []

        self.sentences = []
        self.labels = []

        self.save_to_file = save_to_file
        self.save_file = save_file
        self.n_threads = n_threads
        self.mem_limit = mem_limit

        self.remove_dirty()
        self.build_data()

        if self.save_to_file:
            self.save()

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

    def word_tokenize(self) -> (list, list):
        # Mecab Pos Tagger
        mecab = Mecab()

        def emo(x: str) -> str:
            return emoticon_normalize(x, n_repeats=3)

        def rep(x: str) -> str:
            return repeat_normalize(x, n_repeats=2)

        def normalize(x: str) -> str:
            return rep(emo(x))

        n_data = len(self.data)
        p_data, l_data = [], []
        for idx, d in enumerate(self.data):
            pos = list(map(lambda x: '/'.join(x), mecab.pos(normalize(d['comment']))))

            # append sentence & rate
            p_data.append(pos)
            l_data.append(d['rate'])

            if idx > 0 and idx % (n_data // (100 * self.n_threads)) == 0:
                print("[*] %d/%d" % (idx, n_data), pos)
                gc.collect()
            del pos

        return p_data, l_data

    def build_data(self):
        ts = len(self.data) // self.n_threads  # 5366474
        with Pool(self.n_threads) as p:
            pp_data = [p.apply_async(self.word_tokenize, (self.data[ts * i:ts * (i + 1)],))
                       for i in range(self.n_threads)]

            for d in pp_data:
                self.sentences += d.get()[0]
                self.labels += d.get()[1]

        del self.data
        gc.collect()

    def save(self):
        import unicodecsv as csv

        assert self.save_file

        try:
            with open(fn, 'w', encoding='utf8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['rate', 'comment'])

                w.writeheader()
                for rate, comment in tqdm(zip(self.labels, self.sentences)):
                    w.writerow({'rate': rate, 'comment': ' '.join(comment)})
        except Exception as e:
            raise Exception(e)
