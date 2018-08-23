import gc
import psutil
import pymysql
import logging
import argparse

from tqdm import tqdm
from gensim import corpora
from konlpy.tag import Mecab
from soynlp.normalizer import *
from multiprocessing import Pool
from gensim.models import word2vec, Doc2Vec


# Argument parser
parser = argparse.ArgumentParser(description='Parsing NAVER Movie Review')
parser.add_argument('--n_threads', type=int, help='the number of threads', default=5)
parser.add_argument('--n_mem_limit', type=int, help='ram limitation', default=256)
parser.add_argument('--max_sentences', type=int, help='the number of sentences to train (0: all)', default=2560000)
parser.add_argument('--save_model', type=str, help='trained w2v model file', default='ko_w2v.model')
parser.add_argument('--data_file', type=str, help='movie review data file', default=None)
parser.add_argument('--save_dict', type=bool, help='korean words dictionary', default=False)
parser.add_argument('--load_from', type=str, help='load DataSet from db or csv', default='db')
parser.add_argument('--vector', type=str, help='w2v or d2v', default='d2v')
args = parser.parse_args()

db_infos = {
    'host': 'localhost',
    'user': 'root',
    'password': 'autoset',
    'db': 'movie',
    'charset': 'utf8',
    'cursorclass': pymysql.cursors.DictCursor,
}


vec = args.vector
fn = args.data_file
ko_dict = args.save_dict
load_from = args.load_from
model_name = args.save_model

n_threads = args.n_threads
mem_limit = args.n_mem_limit
max_sentences = args.max_sentences


def get_review_data() -> list:
    db_conn = pymysql.connect(**db_infos)

    data_query = "select rate, comment from movie"
    with db_conn.cursor() as cur:
        cur.execute(data_query)

        rows = cur.fetchall()

    return rows


def to_csv(data: list, fn: str) -> bool:
    import unicodecsv as csv

    try:
        with open(fn, 'w', encoding='utf8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['rate', 'comment'])

            w.writeheader()
            for d in data:
                w.writerow(d)
    except Exception as e:
        raise Exception(e)
    return True


def from_csv(fn: str) -> list:
    data = []
    with open(fn, 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()[1:]):
            d = line.split(',')
            try:
                # remove dirty stuffs
                data.append({'rate': d[0], 'comment': ','.join(d[1:]).replace('\x00', '').replace('\n', '').
                            replace('<span class=""ico_penel""></span>', '').strip('"').strip()})
            except Exception as e:
                print(e, line)
            del d

    if max_sentences == 0:
        return data
    else:
        return data[:max_sentences]


def word_processing(data: list) -> list:
    # Mecab Pos Tagger
    mecab = Mecab()
    
    def emo(x: str) -> str:
        return emoticon_normalize(x, n_repeats=3)

    def rep(x: str) -> str:
        return repeat_normalize(x, n_repeats=2)

    def normalize(x: str) -> str:
        return rep(emo(x))

    idx = 0
    p_data = []
    n_data = len(data)
    for d in data:
        pos = list(map(lambda x: '/'.join(x), mecab.pos(normalize(d['comment']))))
        p_data.append(pos)

        if idx > 0 and idx % (n_data // 100) == 0:
            print("[*] %d/%d" % (idx, n_data), pos)
            gc.collect()
            
            remain_ram = psutil.virtual_memory().available / (2**20)
            if remain_ram < mem_limit:
                import sys
                print("[-] not enough memory < 256MB, ", remain_ram)
                sys.exit(-1)

        del pos
        idx += 1
    return p_data


def w2v_training(data: list) -> bool:
    # word2vec Training
    config = {
        'sentences': data,
        'batch_words': 10000,
        'size': 300,
        'window': 5,
        'min_count': 3,
        'negative': 3,
        'sg': 1,
        'iter': 10,
        'seed': 1337,
        'workers': 8,
    }
    w2v_model = word2vec.Word2Vec(**config)
    w2v_model.wv.init_sims(replace=True)

    w2v_model.save(model_name)
    return True


def d2v_training(data: list) -> bool:
    config = {
        'dm': 1,
        'dm_concat': 1,
        'size': 300,
        'negative': 5,
        'hs': 0,
        'min_count': 2,
        'window': 5,
        'iter': 100,
        'workers': 8,
    }
    d2v_model = Doc2Vec(**config)
    d2v_model.build_vocab(data)

    d2v_model.train(data)

    d2v_model.save(model_name)
    return True


# Getting Review Data from DB/CSV
if load_from == 'db':
    data = get_review_data()
    if fn:  # save to .csv
        to_csv(data, fn)
elif load_from == 'csv':
    data = from_csv(fn)
else:
    raise ValueError("[-] load_from argument must be 'db' or 'csv")

gc.collect()

# Analyze morphs # concat like... word/pos
# To-Do
# * Text Normalizing
# * Text Stemming

datas = []
ts = len(data) // n_threads  # 5366474
with Pool(n_threads) as p:
    pp_data = [p.apply_async(word_processing, (data[ts * i:ts * (i + 1)],)) for i in range(n_threads)]
    
    for d in pp_data:
        datas += d.get()

del data
gc.collect()

# logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if ko_dict:
    corpora.Dictionary(datas).save('ko.dict')

if vec == 'd2v':
    d2v_training(datas)
elif vec == 'w2v':
    w2v_training(datas)  # w2v Training
else:
    raise ValueError("[-] vec should be w2v or d2v")

del datas
gc.collect()

