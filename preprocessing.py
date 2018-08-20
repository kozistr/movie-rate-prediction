import pymysql

from tqdm import tqdm
from soynlp.normalizer import *
from konlpy.tag import Hannanum
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
save_ext = '.'  # '.csv'
load_ext = '.csv'

w2v_model = "./ko_embeddings.model"


def get_review_data() -> list:
    db_conn = pymysql.connect(**db_infos)

    data_query = "select rate, comment from movie"
    with db_conn.cursor() as cur:
        cur.execute(data_query)

        rows = cur.fetchall()

    return rows


def to_json(data: list, fn: str) -> bool:
    import json
    try:
        with open(fn, 'w', encoding='utf8') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False))
    except Exception as e:
        raise Exception(e)
    return True


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
    import csv

    data = []
    with open(fn, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        try:
            for w in tqdm(reader):
                data.append(w['comment'])
        except Exception as e:
            raise Exception(e)
    return data


def word_processing(data: list) -> list:
    global jvm_path

    # Hannanum Pos Tagger
    hannanum = Hannanum(jvmpath=jvm_path)

    def emo(x: str) -> str:
        return emoticon_normalize(x, n_repeats=3)

    def rep(x: str) -> str:
        return repeat_normalize(x, n_repeats=3)

    def normalize(x: str) -> str:
        return rep(emo(x))

    morphs_data = [list(map(lambda x: "/".join(x), hannanum.pos(normalize(d['comment'])))) for d in tqdm(data)]
    return morphs_data


def w2v_training(data: list) -> bool:
    # word2vec Training
    embeddings = word2vec.Word2Vec(sentences=data,
                                   window=5,
                                   max_vocab_size=300,
                                   min_count=5,
                                   negative=3,
                                   seed=1337,
                                   workers=8)

    embeddings.wv.save_word2vec_format(w2v_model, binary=False)
    return True


# Getting Review Data from DB

data = get_review_data()

"""
# Saving into ...
if save_ext == '.csv':
    to_csv(dict_data, fn + save_ext)
elif save_ext == '.json':
    to_json(dict_data, fn + save_ext)
else:
    print("[-] Not Supporting Yet :(")
"""
# data = from_csv(fn + load_ext)

# Analyze morphs # concat like... word/pos
# To-Do
# 1. Text Normalization # https://github.com/open-korean-text/open-korean-text - Done
# 2. Text Tokenization  #
# 3. Text Stemming
m_data = word_processing(data)

for i in range(10):
    print("[*] %d : " % i, m_data[i])

w2v_training(m_data)
