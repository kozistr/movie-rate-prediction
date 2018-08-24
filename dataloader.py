import gc
import psutil
import numpy as np
import csv

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

    def __init__(self, file, save_to_file=False, save_file=None, use_naive_save=False, max_sentences=5000000,
                 n_threads=4, mem_limit=512):
        self.file = file
        assert self.file.find('.csv')

        self.data = []

        self.sentences = []
        self.labels = []

        self.max_sentences = max_sentences + 1

        self.use_naive_save = use_naive_save
        self.save_to_file = save_to_file
        self.save_file = save_file
        self.n_threads = n_threads
        self.mem_limit = mem_limit

        if self.save_file and not self.use_naive_save:
            self.csv_file = open(self.save_file, 'w', encoding='utf8', newline='')

            self.w = csv.DictWriter(self.csv_file, fieldnames=['rate', 'comment'])
            self.w.writeheader()

        self.remove_dirty()
        self.build_data()

        if self.save_to_file:
            if self.use_naive_save:  # if you have a enough memory
                self.naive_save()

    def remove_dirty(self):
        with open(self.file, 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines()[1: self.max_sentences]):
                d = line.split(',')
                try:
                    # remove dirty stuffs
                    self.data.append({'rate': d[0], 'comment': ','.join(d[1:]).replace('\x00', '').replace('\n', '').
                                     replace('<span class=""ico_penel""></span>', '').strip('"').strip()})
                except Exception as e:
                    print(e, line)
                del d

    def word_tokenize(self, data: list) -> (list, list):
        def emo(x: str) -> str:
            return emoticon_normalize(x, n_repeats=3)

        def rep(x: str) -> str:
            return repeat_normalize(x, n_repeats=2)

        def normalize(x: str) -> str:
            return rep(emo(x))

        mecab = Mecab()

        n_data = len(data)
        p_data, l_data = [], []
        for idx, d in enumerate(data):
            pos = list(map(lambda x: '/'.join(x), mecab.pos(normalize(d['comment']))))

            # append sentence & rate
            if not self.use_naive_save:
                self.w.writerow({'rate': d['rate'], 'comment': ' '.join(pos)})

            p_data.append(pos)
            l_data.append(d['rate'])

            if idx > 0 and idx % (n_data // 100) == 0:
                print("[*] %d/%d" % (idx, n_data), pos)

                remain_ram = psutil.virtual_memory().available / (2 ** 20)
                if remain_ram < self.mem_limit:
                    print("[-] not enough memory %dMB < %dMB, " % (remain_ram, self.mem_limit))

                    del pos
                    gc.collect()

                    return p_data, l_data
            del pos
        return p_data, l_data

    def build_data(self):
        ts = len(self.data) // self.n_threads  # 5366474
        with Pool(self.n_threads) as p:
            pp_data = [p.apply_async(self.word_tokenize, (self.data[ts * i:ts * (i + 1)],))
                       for i in range(self.n_threads)]

            for pd in pp_data:
                self.sentences += pd.get()[0]
                self.labels += pd.get()[1]

        del self.data
        gc.collect()

    def naive_save(self):
        assert self.save_file

        try:
            with open(self.save_file, 'w', encoding='utf8', newline='') as csv_file:
                w = csv.DictWriter(csv_file, fieldnames=['rate', 'comment'])

                w.writeheader()
                for rate, comment in tqdm(zip(self.labels, self.sentences)):
                    w.writerow({'rate': rate, 'comment': ' '.join(comment)})
        except Exception as e:
            raise Exception(e)
