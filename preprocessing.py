import time
import json
import pymysql

from multiprocessing import Pool
from soynlp.word import WordExtractor


db_infos = {
    'host': 'localhost',
    'user': 'root',
    'password': 'autoset',
    'db': 'movie',
    'charset': 'utf8',
    'cursorclass': pymysql.cursors.DictCursor,
}

db_conn = None


def db_connection():
    global db_conn

    db_conn = pymysql.connect(**db_infos)


def get_data() -> dict:
    global db_conn

    data_query = "select rate, comment from movie"  # "select comment from movie where rate=%d" % rate
    with db_conn.cursor() as cur:
        start = time.time()
        cur.execute(data_query)
        end = time.time()

        print("[*] Took %ds" % (end - start))

        rows = cur.fetchall()

    return rows


db_connection()

"""
p = Pool(10)
print(p.map(get_data, list(range(1, 11))))

with open('data.json', 'w', encoding='utf8') as f:
    f.write(json.dumps(data))
"""
dict_data = get_data()

with open('data.json', 'w', encoding='utf8') as f:
    f.write(json.dumps(dict_data, ensure_ascii=False))

"""
word_extractor = WordExtractor(min_count=100,
                               min_cohesion_forward=.05,
                               min_right_branching_entropy=.0)
"""
