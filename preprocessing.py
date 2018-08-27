import gc
import pymysql
import logging
import argparse

from tqdm import tqdm
from konlpy.tag import Mecab
from config import get_config

from soynlp.normalizer import *

from multiprocessing import Pool
from collections import namedtuple
from gensim.models import word2vec, Doc2Vec


# Argument parser
parser = argparse.ArgumentParser(description='Pre-Processing NAVER Movie Review Comment')
parser.add_argument('--load_from', type=str, help='load DataSet from db or csv', default='db', choices=['db', 'csv'])
parser.add_argument('--vector', type=str, help='d2v or w2v', choices=['d2v', 'w2v'], default='d2v')
args = parser.parse_args()

config, _ = get_config()  # global configuration

# Korean Pos/Morphs Analyzer
analyzer = None
if config.analyzer == 'mecab':
    from konlpy.tag import Mecab
    analyzer = Mecab()
elif config.analyzer == 'hannanum':
    from konlpy.tag import Hannanum
    analyzer = Hannanum(jvmpath=config.jvm_path)
elif config.analyzer == 'twitter':  # Not Recommended
    from konlpy.tag import Twitter
    analyzer = Twitter(jvmpath=config.jvm_path)

# Korean Word Spacing
if config.use_correct_spacing:
    try:
        from pykospacing import spacing
    except Exception as e:
        raise Exception(e)

vec = args.vector
load_from = args.load_from


def from_db(db_cfg) -> list:
    db_info = {
        'host': db_cfg.host,
        'user': db_cfg.user,
        'password': db_cfg.password,
        'db': db_cfg.db,
        'charset': db_cfg.charset,
        'cursorclass': pymysql.cursors.DictCursor,
    }
    db_conn = pymysql.connect(**db_info)

    with db_conn.cursor() as cur:
        cur.execute("select rate, comment from movie")
        rows = cur.fetchall()
    return rows


def from_csv(fn_csv: str) -> list:
    csv_data = []
    with open(fn_csv, 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()[1:]):
            dl = line.split(',')
            try:
                csv_data.append({'rate': dl[0], 'comment': ','.join(dl[1:])})
            except Exception as E:
                print(E, line)
            del dl
    return csv_data


def to_csv(w_data: list, fn_csv: str) -> bool:
    try:
        import unicodecsv as csv

        with open(fn_csv, 'w', encoding='utf8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['rate', 'comment'])
            w.writeheader()
            for d_ in w_data:
                w.writerow(d_)
    except Exception as E:
        raise Exception(E)
    return True


def word_cleaning(w_data: list) -> list:
    w_len = len(w_data)
    for w_idx in tqdm(range(w_len)):  # faster way than w_data iter
        w_data[w_idx]['comment'] = w_data[w_idx]['comment'].replace('<span class=""ico_penel""></span>', '').\
            replace('\x00', '').replace('\n', '').strip('"').strip()
    return w_data


def word_processing(w_data: list) -> (list, list):
    global config

    def emo(x: str) -> str:
        return emoticon_normalize(x, n_repeats=3)

    def rep(x: str) -> str:
        return repeat_normalize(x, n_repeats=2)

    def normalize(x: str) -> str:
        return rep(emo(x)) if config.use_normalize else x

    w_len = len(w_data)
    p_data, l_data = [], []
    for w_idx in tqdm(range(w_len)):  # faster way than w_data iter:
        pos = list(map(lambda x: '/'.join(x), analyzer.pos(normalize(w_data[w_idx]['comment']))))

        # append sentence & rate
        p_data.append(pos)
        l_data.append(w_data[w_idx]['rate'])

        if w_idx > 0 and w_idx % (w_len // (100 * config.n_threads)) == 0:
            print("[*] %d/%d" % (w_idx, w_len), pos)
            gc.collect()
        del pos
    return p_data, l_data


def w2v_training(data: list) -> bool:
    global config

    # word2vec Training
    config = {
        'sentences': data,
        'batch_words': 10000,
        'size': config.embed_size,
        'window': 5,
        'min_count': 3,
        'negative': 3,
        'sg': 1,
        'iter': 10,
        'seed': config.seed,
        'workers': config.n_threads,
    }
    w2v_model = word2vec.Word2Vec(**config)
    w2v_model.wv.init_sims(replace=True)

    w2v_model.save(config.w2v_model)
    return True


def d2v_training(sentences: list, rates: list, epochs=10) -> bool:
    global config

    # data processing to fit in Doc2Vec
    taggedDocs = namedtuple('TaggedDocument', 'words tags')
    tagged_data = [taggedDocs(s, r) for s, r in zip(sentences, rates)]

    config = {
        'dm': 1,
        'dm_concat': 1,
        'vector_size': config.embed_size,
        'negative': 3,
        'hs': 0,
        'alpha': config.lr,
        'min_alpha': config.min_lr,
        'min_count': 3,
        'window': 5,
        'seed': config.seed,
        'workers': config.n_threads,
    }
    d2v_model = Doc2Vec(**config)
    d2v_model.build_vocab(tagged_data)

    total_examples = len(sentences)
    for _ in tqdm(range(epochs)):
        # Doc2Vec training
        d2v_model.train(tagged_data, total_examples=total_examples, epochs=d2v_model.iter)

        # LR Scheduler
        d2v_model.alpha -= config.lr_decay
        d2v_model.min_alpha = d2v_model.alpha

    d2v_model.save(config.d2v_model)
    return True


def main():
    # Stage 1 : Parsing Data from DB or .csv(file)
    if load_from == 'db':
        raw_data = from_db(config)
    else:
        raw_data = from_csv(config)

    # Stage 2 : Cleaning Data
    cleaned_data = word_cleaning(raw_data)
    del raw_data  # release unused memory

    # Stage 3 : word processing
    x_data, y_data = word_processing(cleaned_data)

    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if vec == 'd2v':
        d2v_training(x_data, y_data)  # d2v Training
    elif vec == 'w2v':
        w2v_training(x_data)          # w2v Training

    del x_data
    del y_data
    gc.collect()


if __name__ == "__main__":
    main()

"""
# Multi-Threading implementation
x_data, y_data = [], []
ts = len(data) // n_threads  # 5366474
with Pool(n_threads) as p:
    pp_data = [p.apply_async(word_processing, (data[ts * i:ts * (i + 1)],)) for i in range(config.n_threads)]
    
    for d in pp_data:
        x_data += d.get()[0]
        y_data += d.get()[1]
"""