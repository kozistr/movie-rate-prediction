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
}

data = dict()
db_conn = None


def db_connection():
    global db_conn

    db_conn = pymysql.connect(host=db_infos['host'],
                              user=db_infos['user'],
                              password=db_infos['password'],
                              db=db_infos['db'],
                              charset=db_infos['charset'],
                              cursorclass=pymysql.cursors.DictCursor
                              )


def get_data(rate: int) -> bool:
    global db_conn, data

    data_query = "select comment from movie where rate=%d" % rate
    with db_conn.cursor() as cur:
        start = time.time()
        cur.execute(data_query)
        end = time.time()

        print("[*] Took %ds" % (end - start))

        for row in cur:
            data.update({"rate": rate, "comment": row})

    return True


db_connection()

p = Pool(10)
print(p.map(get_data, list(range(1, 11))))

with open('data.json', 'w', encoding='utf8') as f:
    f.write(json.dumps(data))

"""
word_extractor = WordExtractor(min_count=100,
                               min_cohesion_forward=.05,
                               min_right_branching_entropy=.0)
"""
