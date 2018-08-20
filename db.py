import sys
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
conn = pymysql.connect(host=db_infos['host'],
                       user=db_infos['user'],
                       password=db_infos['password'],
                       charset=db_infos['charset']
                       )

# Delete previous database
with conn.cursor() as cur:
    try:
        database_query = "drop schema movie"
        cur.execute(database_query)
        conn.commit()
    except pymysql.Warning as w:
        pass
    except pymysql.err.InternalError as e:
        pass  # Database doesn't exist
    except pymysql.err.ProgrammingError as e:
        raise pymysql.err.ProgrammingError(e)

# Make new 'movie' database
with conn.cursor() as cur:
    try:
        database_query = "create database movie"
        cur.execute(database_query)
        conn.commit()
    except pymysql.Warning as w:
        pass
    except pymysql.err.InternalError as e:
        pass  # Database exists
    except pymysql.err.ProgrammingError as e:
        raise pymysql.err.ProgrammingError(e)

with conn.cursor() as cur:
    try:
        with open(query_paths['table_query'], 'r') as f:
            table_query = f.read()

        # Making 'movie' table
        conn.cursor().execute(table_query)
        conn.commit()
    except pymysql.Warning as w:
        pass
    except Exception as e:
        raise Exception(e)

conn.close()

# New DB Connection
conn = pymysql.connect(**db_infos)

n_query, n_success = 0, 0
with conn.cursor() as cur:
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
            print("[-]", qp, e)

            if e.args[0] == 1146 or e.args[0] == 1046:  # Table/Database Doesn't exist
                sys.exit(-1)

        n_query += 1

print("[+] Total %d/%d Queries Success!" % (n_success, n_query))

# Close DB Session
conn.close()
