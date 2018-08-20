import pymysql

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


def from_csv(fn: str) -> dict:
    import unicodecsv as csv

    data = dict()
    try:
        with open(fn, 'r', encoding='utf8') as f:
            for w in csv.DictReader(f):
                data.update(w)
    except Exception as e:
        raise Exception(e)
    return data


# Getting Review Data from DB
dict_data = get_review_data()

# Saving into ...
if save_ext == '.csv':
    to_csv(dict_data, fn + save_ext)
elif save_ext == '.json':
    to_json(dict_data, fn + save_ext)
else:
    print("[-] Not Supporting Yet :(")

# Hannanum Pos Tagger
hannanum = Hannanum(jvmpath=jvm_path)

# Analyze morphs # concat like... word/pos
# To-Do
# 1. Text Normalization # https://github.com/open-korean-text/open-korean-text - Done
# 2. Text Tokenization  #
# 3. Text Stemming

emo_f = lambda x: emoticon_normalize(x, n_repeats=3)
rep_f = lambda x: repeat_normalize(x, n_repeats=2)
morphs_data = [list(map(lambda x: "/".join(x), rep_f(emo_f(d['comment'])))) for d in dict_data]

# word2vec Training
embeddings = word2vec.Word2Vec(sentences=morphs_data,
                               window=5,
                               max_vocab_size=300,
                               min_count=5,
                               negative=3,
                               workers=8)

embeddings.save_word2vec_format('korean_embeddings.model', binary=False)
