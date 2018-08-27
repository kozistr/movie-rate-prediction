import sys
import pymysql

from glob import glob
from tqdm import tqdm
from config import get_config


def make_db_conn(db_info: dict):
    db_conn = pymysql.connect(**db_info)
    return db_conn


def do_db(db_conn, qry: str):
    with db_conn.cursor() as db_cur:
        try:
            db_cur.execute(qry)
            db_conn.commit()
        except pymysql.Warning:
            pass
        except pymysql.err.InternalError:
            pass  # Database doesn't exist
        except pymysql.err.ProgrammingError as pe:
            raise pymysql.err.ProgrammingError(pe)
        except Exception as E:
            print("[-]", qry, E)

            if E.args[0] == 1146 or E.args[0] == 1046:  # Table/Database Doesn't exist
                sys.exit(-1)
        return True


def main():
    # get configuration
    cfg, _ = get_config()

    db_info = {
        'host': cfg.host,
        'user': cfg.user,
        'password': cfg.password,
        'charset': cfg.charset,
    }

    # Stage 1 : Initial DB Connection
    print("[*] Making DB Connection...")
    db_con = make_db_conn(db_info)

    # Stage 2 : Delete previous database
    print("[*] Drop 'movie' database")
    do_db(db_con, "drop schema movie")

    # Stage 3 : Make 'movie' database
    print("[*] Create 'movie' database")
    do_db(db_con, "create database movie")

    # Stage 4 : Make 'movie.movie' table
    print("[*] Create 'movie.movie' table")
    with open('table.sql', 'r') as table_qry:
        do_db(db_con, table_qry.read())

    db_con.close()  # close db

    db_info['db'] = 'movie'         # update db info
    db_con = make_db_conn(db_info)  # db connect

    # Stage 5 : Insert whole queries
    print("[*] Inserting data queries...")
    n_query, n_success = 0, 0
    for qp in tqdm(glob(cfg.query_path + "*.sql")):
        item_qry = open(qp, 'r', encoding='utf8').read()

        # insert item into movie.movie
        if do_db(db_con, item_qry):
            n_success += 1
        n_query += 1

    print("[+] Total %d/%d Queries Success!" % (n_success, n_query))
    db_con.close()


if __name__ == '__main__':
    main()
