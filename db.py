import pymysql

from glob import glob
from tqdm import tqdm


query_paths = {
    'data_query': "./comments/*.sql",
    'table_query': "./table.sql",
}

db_infos = {
    'host': 'localhost',
    'user': 'root',
    'password': 'autoset',
    'db': 'movie',
    'charset': 'utf8',
}


# MySQL DB Connection
try:
    conn = pymysql.connect(host=db_infos['host'],
                           user=db_infos['user'],
                           password=db_infos['password'],
                           db=db_infos['db'],
                           charset=db_infos['charset']
                           )

except Exception as e:  # If there's no 'movie' DB
    print(e)

    conn = pymysql.connect(host=db_infos['host'],
                           user=db_infos['user'],
                           charset=db_infos['charset']
                           )

    with conn.cursor() as cur:
        # Making 'movie' DB
        database_query = "create database movie"
        cur.execute(database_query)
        conn.commit()

        with open(query_paths['table_query'], 'r') as f:
            table_query = f.read()

        # Making 'movie' table
        cur.execute(table_query)
        conn.commit()

cur = conn.cursor()

n_query, n_success = 0, 0
for qp in tqdm(glob(query_paths['data_query'])):
    # Read SQL Query
    with open(qp, 'r', encoding='utf8') as f:
        query = f.read()

    try:
        cur.execute(query=query)  # Execute SQL Query
        conn.commit()             # Commit Changes

        n_success += 1
    except pymysql.Warning as w:
        pass  # Most of warnings are about "Data truncated ~~"
    except Exception as e:
        print("[-] ", qp, " ", e)

    n_query += 1

print("[+] Total %d/%d Queries Success!" % (n_success, n_query))

# Close DB Session
conn.close()
