import gc
import psutil
import pymysql
import logging

from tqdm import tqdm
from gensim import corpora
from konlpy.tag import Mecab
from soynlp.normalizer import *
from multiprocessing import Pool
from gensim.models import word2vec


db_infos = {
    'host': 'localhost',
    'user': 'root',
    'password': 'autoset',
    'db': 'movie',
    'charset': 'utf8',
    'cursorclass': pymysql.cursors.DictCursor,
}

jvm_path = "C:\\Program Files\\Java\\jre-9\\bin\\server\\jvm.dll"

fn = "data"
ext = '.csv'
w2v_model_name = "ko_embeddings.model"

n_threads = 5
mem_limit = 256  # 256MB
max_sentences = 2500000


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
                data.append({'rate': d[0], 'comment': ','.join(d[1:]).replace('\x00', '').replace('\xa0', '').replace('\n', '').strip('"')})
            except Exception as e:
                print(e, line)
            del d

    if max_sentences == 0:
        return data
    else:
        return data[:max_sentences]


def word_processing(data: list) -> list:
    global jvm_path

    # Hannanum Pos Tagger
    # hannanum = Hannanum()
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

        if idx == n_data - 1:
            for idx, p in enumerate(p_data[:3]):
                print("[*] %d" % idx, p)

        del pos
        idx += 1

    print("[+] Done!")
    return p_data


def w2v_training(data: list) -> bool:
    # loggin
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    dictionary_ko = corpora.Dictionary(data)
    dictionary_ko.save('ko.dict')
    
    # word2vec Training
    config = {
        'sentences': data,
        'batch_words': 10000,
        'size': 300,
        'window': 5,
        'min_count': 2,
        'negative': 3,
        'sg': 1,
        'iter': 10,
        'downsample': 1e-3,
        'seed': 1337,
        'workers': 8,
    }
    w2v_model = word2vec.Word2Vec(**config)
    w2v_model.wv.init_sims(replace=True)
    
    w2v_model.save('ko_w2v.model')
    w2v_model.wv.save_word2vec_format(w2v_model_name, binary=False)
    return True


# Getting Review Data from DB
"""
data = get_review_data()

to_csv(data, fn + ext)
"""
data = from_csv(fn + ext)

gc.collect()

# Analyze morphs # concat like... word/pos
# To-Do
# 1. Text Normalization # https://github.com/open-korean-text/open-korean-text - Done
# 2. Text Tokenization  #
# 3. Text Stemming

ts = len(data) // n_threads  # 5366474

datas = []
with Pool(n_threads) as p:
    pp_data = [p.apply_async(word_processing, (data[ts * i:ts * (i + 1)],)) for i in range(n_threads)]
    
    for d in pp_data:
        datas += d.get()
 
del data
gc.collect()
 
print("[*] Total data : %d" % len(datas))   

for i in range(5):
    print("[*] %d : " % i, datas[i])
        
w2v_training(datas)

del datas
gc.collect()

