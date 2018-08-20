import pymysql

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

java_home = "C:\\Program Files\\Java\\jre-9\\bin\\server\\jvm.dll"

fn = "data"
save_ext = '.csv'


def get_review_data() -> dict:
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


# Getting Review Data from DB
dict_data = get_review_data()

# Saving into ...
if save_ext == '.csv':
    to_csv(dict_data, fn + save_ext)
elif save_ext == '.json':
    to_json(dict_data, fn + save_ext)
else:
    print("[-] Not Supporting Yet :(")

