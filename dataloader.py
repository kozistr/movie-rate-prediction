import gc
import csv
import psutil
import numpy as np

from tqdm import tqdm
from konlpy.tag import Mecab
from soynlp.normalizer import *
from pykospacing import spacing
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

    def sent_to_vec(self, input_sentence: str) -> np.array:
        return self.d2v_model.infer_vector(input_sentence)

    def __len__(self):
        return len(self.d2v_model.wv.vocab)


class DataLoader:

    def __init__(self, file, is_tagged_file=False, save_to_file=False, save_file=None, use_in_time_save=True,
                 max_sentences=-2, n_threads=5, mem_limit=512):
        self.file = file

        self.data = []
        self.sentences = []
        self.labels = []

        self.max_sentences = max_sentences + 1

        self.is_tagged_file = is_tagged_file
        self.use_in_time_save = use_in_time_save
        self.save_to_file = save_to_file
        self.save_file = save_file
        self.n_threads = n_threads  # currently, unsupported feature :(
        self.mem_limit = mem_limit

        self.mecab = Mecab()  # Korean Pos Tagger

        assert self.file.find('.csv')
        assert self.save_file

        if self.use_in_time_save:
            self.csv_file = open(self.save_file, 'w', encoding='utf8', newline='')
            self.csv_file.writelines("rate,comment\n")  # csv header
            print("[*] %s is generated!" % self.save_file)

        if not self.is_tagged_file:
            # Stage 1 : remove dirty stuffs / normalizing
            self.remove_dirty()

            # Stage 2 : build the data (word processing)
            self.build_data()

            if not self.use_in_time_save:  # if you have a enough memory
                self.naive_save()
        else:
            self.naive_load()  # just load from .csv

    def remove_dirty(self, sent_spacing=False):
        with open(self.file, 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines()[1: self.max_sentences]):
                d = line.split(',')
                try:
                    # remove dirty stuffs
                    if sent_spacing:
                        self.data.append({'rate': d[0],
                                          'comment': spacing(','.join(d[1:]).replace('\x00', '').replace('\n', '').
                                                             replace('<span class=""ico_penel""></span>', '').
                                                             strip('"').strip())})
                    else:
                        self.data.append({'rate': d[0],
                                          'comment': ','.join(d[1:]).replace('\x00', '').replace('\n', '').
                                         replace('<span class=""ico_penel""></span>', '').strip('"').strip()})
                except Exception as e:
                    print(e, line)
                del d

    def word_tokenize(self):
        def emo(x: str) -> str:
            return emoticon_normalize(x, n_repeats=3)

        def rep(x: str) -> str:
            return repeat_normalize(x, n_repeats=2)

        def normalize(x: str) -> str:
            return rep(emo(x))

        n_data = len(self.data)
        for idx, cd in enumerate(self.data):
            pos = list(map(lambda x: '/'.join(x), self.mecab.pos(normalize(cd['comment']))))

            if self.use_in_time_save:
                self.csv_file.writelines(str(cd['rate']) + ',' + ' '.join(pos) + '\n')

            self.sentences.append(pos)
            self.labels.append(cd['rate'])

            if idx > 0 and idx % (n_data // 100) == 0:
                print("[*] %d/%d" % (idx, n_data), pos)

                remain_ram = psutil.virtual_memory().available / (2 ** 20)
                if remain_ram < self.mem_limit:
                    raise MemoryError("[-] not enough memory %dMB < %dMB, " % (remain_ram, self.mem_limit))
            del pos
        gc.collect()

    def build_data(self):
        """
            ts = len(self.data) // self.n_threads  # 5366474
            with Pool(self.n_threads) as pool:
                print(pool.map(self.word_tokenize, [self.data[ts * i:ts * (i + 1)] for i in range(self.n_threads)]))
    
                pp_data = [pool.apply_async(self.word_tokenize, (self.data[ts * i:ts * (i + 1)],))
                           for i in range(self.n_threads)]
               
                for pd in pp_data:
                    self.sentences += pd.get()[0]
                    self.labels += pd.get()[1]
        """

        self.word_tokenize()

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

    def naive_load(self):
        with open(self.file, 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines()[1:]):
                d = line.split(',')

                self.sentences.append(d[1].split(' '))
                self.labels.append(d[0])
