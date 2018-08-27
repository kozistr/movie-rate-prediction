import gc
import csv
import numpy as np

from tqdm import tqdm
from soynlp.normalizer import *


class Word2VecEmbeddings:

    def __init__(self, w2v_model, dims=300):
        self.model = w2v_model

        self.dims = dims

        self.w2v_model = None
        self.embeds = None

        self.load_model()
        self.build_embeds()

    def load_model(self):
        from gensim.models import Word2Vec
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
        from gensim.models import Doc2Vec
        self.d2v_model = Doc2Vec.load(self.model)

    def sent_to_vec(self, input_sentence: str) -> np.array:
        return self.d2v_model.infer_vector(input_sentence)

    def __len__(self):
        return len(self.d2v_model.wv.vocab)


class DataLoader:

    def __init__(self, file, n_classes=10, analyzer='mecab',
                 use_correct_spacing=False, use_normalize=True,
                 load_from='db', is_analyzed=False, fn_to_save=None, use_save=True, jvm_path=None,
                 config=None):
        self.file = file
        self.n_classes = n_classes

        self.data = []

        self.sentences = []
        self.labels = []

        self.use_correct_spacing = use_correct_spacing
        self.use_normalize = use_normalize

        self.is_analyzed = is_analyzed
        self.fn_to_save = fn_to_save
        self.load_from = load_from
        self.use_save = use_save
        self.jvm_path = jvm_path

        self.analyzer = analyzer

        self.config = config

        assert self.config
        assert self.file.find('.csv') and self.fn_to_save
        assert not self.analyzer == 'mecab' and self.jvm_path
        assert self.load_from == 'db' or self.load_from == 'csv'

        if self.analyzer == 'mecab':
            from konlpy.tag import Mecab
            self.analyzer = Mecab()
        elif self.analyzer == 'hannanum':
            from konlpy.tag import Hannanum
            self.analyzer = Hannanum(jvmpath=self.jvm_path)
        elif self.analyzer == 'twitter':
            from konlpy.tag import Twitter
            self.analyzer = Twitter(jvmpath=self.jvm_path)
        else:
            # if is_analyzed is True, there's no need to analyze again.
            if not self.is_analyzed:
                raise NotImplementedError("[-] only Mecab, Hannanum, Twitter are supported :(")

        if self.use_save:
            self.csv_file = open(self.fn_to_save, 'w', encoding='utf8', newline='')
            self.csv_file.writelines("rate,comment\n")  # csv header
            print("[*] %s is generated!" % self.fn_to_save)

        # Already Analyzed Data
        if self.is_analyzed:
            self.naive_load()  # just load data from .csv
        else:
            # Stage 1 : read data from 'db' or 'csv'
            if self.load_from == 'db':
                self.read_from_db()
            else:
                self.read_from_csv()

            # Stage 2-1 : remove dirty stuffs
            self.words_cleaning()

            # Stage 2-2 : (Optional) Correcting spacing
            if self.use_correct_spacing:
                self.correct_spacing()

            # Stage 3 : build data (pos/morphs analyze)
            self.word_tokenize()

            del self.data  # remove unused var # for saving memory
            gc.collect()

        # if it's not binary class, convert into one-hot vector
        if not self.n_classes == 1:
            self.to_one_hot()

    def read_from_db(self):
        import pymysql

        db_info = {
            'host': self.config.host,
            'user': self.config.user,
            'password': self.config.password,
            'db': self.config.db,
            'charset': self.config.charset,
            'cursorclass': pymysql.cursors.DictCursor,
        }
        db_conn = pymysql.connect(**db_info)

        with db_conn.cursor() as cur:
            cur.execute("select rate, comment from movie")
            self.data = cur.fetchall()

    def read_from_csv(self):
        with open(self.file, 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines()[1:]):
                d = line.split(',')
                try:
                    self.data.append({'rate': d[0], 'comment': ','.join(d[1:])})
                except Exception as e:
                    print(e, line)
                del d

    def words_cleaning(self):
        len_data = len(self.data)
        for idx in tqdm(range(len_data)):
            self.data[idx]['comment'] = self.data[idx]['comment'].replace('<span class=""ico_penel""></span>', '').\
                replace('\x00', '').replace('\n', '').strip('"').strip()

    def correct_spacing(self):
        try:
            from pykospacing import spacing
        except ImportError:
            raise ImportError("[-] plz installing KoSpacing package first!")

        len_data = len(self.data)
        for idx in tqdm(range(len_data)):
            self.data[idx]['comment'] = spacing(self.data[idx]['comment'])

    def word_tokenize(self):
        def emo(x: str, n_rep: int = 3) -> str:
            return emoticon_normalize(x, n_repeats=n_rep)

        def rep(x: str, n_rep: int = 3) -> str:
            return repeat_normalize(x, n_repeats=n_rep)

        def normalize(x: str, n_rep: int = 3) -> str:
            return rep(emo(x, n_rep), n_rep) if self.use_normalize else x

        len_data = len(self.data)
        for idx, d in tqdm(enumerate(self.data)):
            pos = list(map(lambda x: '/'.join(x), self.analyzer.pos(normalize(d['comment']))))

            if self.use_save:
                self.csv_file.writelines(str(d['rate']) + ',' + ' '.join(pos) + '\n')

            self.sentences.append(pos)
            self.labels.append(d['rate'])

            if idx > 0 and idx % (len_data // 100) == 0:
                print("[*] %d/%d" % (idx, len_data), pos)
            del pos

    def naive_save(self):
        assert self.fn_to_save

        try:
            with open(self.fn_to_save, 'w', encoding='utf8', newline='') as csv_file:
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

    def to_one_hot(self):
        arr = np.eye(self.n_classes)
        for i in tqdm(range(len(self.labels))):
            self.labels[i] = arr[self.labels[i] - 1]  # 1 ~ 10


class DataIterator:

    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y

        self.batch_size = batch_size
        self.num_examples = num_examples = x.shape[0]
        self.num_batches = num_examples // batch_size
        self.pointer = 0

        assert (self.batch_size <= self.num_examples)

    def next_batch(self):
        start = self.pointer
        self.pointer += self.batch_size

        if self.pointer > self.num_examples:
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)

            self.x = self.x[perm]
            self.y = self.y[perm]

            start = 0
            self.pointer = self.batch_size

        end = self.pointer

        return self.x[start:end], self.y[start:end]

    def iterate(self):
        for step in range(self.num_batches):
            yield self.next_batch()
