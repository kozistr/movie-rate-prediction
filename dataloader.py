import gc
import csv
import h5py
import numpy as np

from tqdm import tqdm
from soynlp.normalizer import *
from bs4 import BeautifulSoup as bs


class Word2VecEmbeddings:

    def __init__(self, w2v_model, dims=300):
        self.model = w2v_model

        self.dims = dims

        self.w2v_model = None
        self.embeds = None

        self.load_model()

        self.vocab_size = len(self.w2v_model.wv.vocab) + 1  # 1 for zero embedding for unknown

        self.build_embeds()

    def load_model(self):
        from gensim.models import Word2Vec
        self.w2v_model = Word2Vec.load(self.model)

    def build_embeds(self):
        self.embeds = np.zeros((self.vocab_size, self.dims))

        for i in tqdm(range(self.vocab_size - 1)):
            vec = self.w2v_model.wv[self.w2v_model.wv.index2word[i]]
            if vec is not None:
                self.embeds[i] = vec

        # zero embedding
        self.embeds[self.vocab_size - 1] = np.zeros((1, self.dims))

    def word_to_vec(self, input_word: str) -> np.array:
        return self.w2v_model.wv[input_word]

    def words_to_index(self, input_words: list) -> list:
        return [self.w2v_model.wv.vocab[word].index if word in self.w2v_model.wv.vocab else self.vocab_size - 1
                for word in input_words]

    def __len__(self):
        return len(self.w2v_model.wv.vocab) + 1

    def __str__(self):
        return "Word2Vec"


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

    def __str__(self):
        return "Doc2Vec"


class EmbeddingVectorLoader:

    def __init__(self, vec=None, n_dims=300, vec_type='tf-idf',
                 save_to_h5=None, load_from_h5=False,
                 config=None):
        self.x_data = None
        self.y_data = None

        self.vec = vec
        self.n_dims = n_dims
        self.vec_type = vec_type

        self.to_vec = vec.word_to_vec if str(vec) == 'Word2Vec' else vec.sent_to_vec

        self.save_to_h5 = save_to_h5
        self.load_from_h5 = load_from_h5

        assert self.vec

        if self.vec_type == 'tf-idf':
            self.vec_type = self.tf_idf_embedding
        elif self.vec_type == 'average':
            self.vec_type = self.mean_embedding
        else:
            raise NotImplementedError("[-] Only tf-idf and average")

        if not self.load_from_h5:
            ds = DataLoader(file=config.processed_dataset,
                            n_classes=config.n_classes,
                            analyzer=None,
                            is_analyzed=True,
                            use_save=False,
                            config=config)  # DataSet Loader
            ds_len = len(ds)

            if config.verbose:
                print("[+] DataSet loaded! Total %d samples" % ds_len)

            # words Vectorization # type conversion
            self.y_data = np.zeros((ds_len, config.n_classes), dtype=np.uint8)

            if config.use_pre_trained_embeds == 'd2v':
                self.x_data = np.zeros((ds_len, config.embed_size), dtype=np.float32)

                for idx in tqdm(range(ds_len)):
                    self.x_data[idx] = self.to_vec(ds.sentences[idx])
                    self.y_data[idx] = np.asarray(ds.labels[idx])
                    ds.sentences[idx] = None
                    ds.labels[idx] = None
            else:
                self.x_data = self.vec_type(ds.sentences)
                del ds.sentences

                for idx in tqdm(range(ds_len)):
                    self.y_data[idx] = np.asarray(ds.labels[idx])
                    ds.labels[idx] = None

            if config.verbose:
                print("[+] conversion finish! x_data, y_data loaded!")

            # delete DataSetLoader() from memory
            ds = None
            gc.collect()

            if self.save_to_h5:
                print("[*] start writing .h5 file...")
                with h5py.File(self.save_to_h5, 'w') as h5fs:
                    h5fs.create_dataset('comment', data=np.array(self.x_data))
                    h5fs.create_dataset('rate', data=np.array(self.y_data))

                if config.verbose:
                    print("[+] data saved into h5 file!")
        else:
            with h5py.File(self.load_from_h5, 'r') as f:
                self.x_data = np.array(f['comment'], dtype=np.float32)
                self.y_data = np.array(f['rate'], dtype=np.uint8)

                if not self.y_data.shape[1] == config.n_classes:
                    print("[*] different 'n_classes' is detected with config file")
                    new_y_data = np.zeros((len(self.y_data), config.n_classes), dtype=np.uint8)

                    arr = np.eye(config.n_classes)
                    for i in tqdm(range(len(self.y_data))):
                        new_y_data[i] = arr[int(self.y_data[i]) - 1] if not config.n_classes == 1 \
                            else np.argmax(self.y_data[i], axis=-1) + 1

                    self.y_data = new_y_data[:]
                    del new_y_data, arr

                if config.verbose:
                    print("[+] data loaded from h5 file!")
                    print("[*] comment : ", self.x_data.shape)
                    print("[*] rate    : ", self.y_data.shape)

    def mean_embedding(self, sentences: list) -> np.array:
        return np.array([
            np.mean([self.to_vec[word] for word in sentence if self.to_vec[word]] or [np.zeros(self.n_dims)], axis=0)
            for sentence in tqdm(sentences)
        ])

    def tf_idf_embedding(self, sentences: list) -> np.array:
        from collections import defaultdict
        from sklearn.feature_extraction.text import TfidfVectorizer

        tf_idf = TfidfVectorizer(analyzer=lambda x: x)
        tf_idf.fit(sentences)

        max_idf = max(tf_idf.idf_)
        word2weight = defaultdict(
            lambda: max_idf,
            [(w, tf_idf.idf_[i]) for w, i in tf_idf.vocabulary_.items()]
        )

        return np.array([
            np.mean([self.to_vec[word] * word2weight[word] for word in sentence if self.to_vec[word]] or
                    [np.zeros(self.n_dims)], axis=0) for sentence in tqdm(sentences)
        ])


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
        self.max_sent_len = 0

        self.use_correct_spacing = use_correct_spacing
        self.use_normalize = use_normalize

        self.is_analyzed = is_analyzed
        self.fn_to_save = fn_to_save
        self.load_from = load_from
        self.use_save = use_save
        self.jvm_path = jvm_path

        self.analyzer = analyzer

        self.config = config

        # Sanity Checks
        assert self.config
        if not self.load_from == 'db':
            assert not self.file.find('.csv') == -1
        if self.use_save:
            assert self.fn_to_save
        if self.analyzer and not self.analyzer == 'mecab':
            assert self.jvm_path

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

        # Already Analyzed Data
        if self.is_analyzed:
            self.naive_load()  # just load data from .csv
        else:
            # Stage 1 : read data from 'db' or 'csv'
            print("[*] loaded from %s" % self.load_from)
            if self.load_from == 'db':
                self.read_from_db()
            else:
                self.read_from_csv()  # currently unstable...

            # Stage 2-1 : remove dirty stuffs
            print("[*] cleaning words...")
            self.words_cleaning()

            # Stage 2-2 : (Optional) Correcting spacing
            if self.use_correct_spacing:
                print("[*] correcting spacing problem...")
                self.correct_spacing()

            if self.use_save:
                self.csv_file = open(self.fn_to_save, 'w', encoding='utf8', newline='')
                self.csv_file.writelines("rate,comment\n")  # csv header
                print("[*] %s is generated!" % self.fn_to_save)

            # Stage 3 : build data (pos/morphs analyze)
            print("[*] start the analyzer")
            self.word_tokenize()

            del self.data  # remove unused var # for saving memory
            gc.collect()

        # if it's not binary class, convert into one-hot vector
        if not self.n_classes == 1:
            self.labels = self.to_one_hot(self.labels, self.n_classes)

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
            csv_f = csv.reader(f)

            idx = 0
            for line in tqdm(csv_f):
                if idx == 0:
                    idx += 1
                    continue

                self.data.append({'rate': line[0], 'comment': bs(line[1], 'lxml').text})
                idx += 1

    def words_cleaning(self):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

        import validators

        drop_list = []
        len_data = len(self.data)
        for idx in tqdm(range(len_data)):
            self.data[idx]['comment'] = bs(self.data[idx]['comment'], "lxml").text.replace('\x00', '').\
                replace('\n', '').strip('"').strip()

            # There're lots of meaningless comments like url... So, I'll drop it from data
            if validators.url(self.data[idx]['comment']):
                drop_list.append(idx)

        print("[*] deleting data which contains only meaningless url")
        for drop in tqdm(sorted(drop_list, reverse=True)):
            del self.data[drop]

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

            if idx and idx % (len_data // 100) == 0:
                print("[*] %d/%d" % (idx, len_data), pos)
            del pos

        self.csv_file.close()

    def naive_save(self):
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
            if self.config.verbose:
                print("[*] %s loaded!" % self.file)

            for line in tqdm(f.readlines()[1:]):
                d = line.split(',')
                try:
                    sent = d[1].split(' ')
                    if len(sent) > self.max_sent_len:
                        self.max_sent_len = len(sent)

                    self.sentences.append(sent)
                    self.labels.append(d[0])
                except IndexError:
                    print("[-] ", line)

        if self.config.verbose:
            print("[*] the number of words in sentence : %d" % self.max_sent_len)

    @staticmethod
    def to_one_hot(data, n_classes: int):
        arr = np.eye(n_classes)
        for i in tqdm(range(len(data))):
            data[i] = arr[int(data[i]) - 1]  # 1 ~ 10
        return data

    @staticmethod
    def to_binary(data):
        for i in tqdm(range(len(data))):
            data[i] = np.argmax(data[i], axis=-1) + 1
        return data

    def __len__(self):
        return len(self.sentences)


class DataIterator:

    def __init__(self, x, y, batch_size):
        # x, y should be numpy obj
        assert not isinstance(x, list) and not isinstance(y, list)

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
