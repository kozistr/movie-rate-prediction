# Credited by 'bab2min', Reformatted by 'kozstr'
#  original link : http://bab2min.tistory.com/556

from concurrent.futures import ThreadPoolExecutor
from config import get_config
from tqdm import tqdm

import urllib
import urllib.parse
import urllib.request

import re
import os
import bs4
import time


# get configuration
cfg, _ = get_config()


def get_comments(code):
    def make_args(c, p):
        params = {
            'code': c,
            'type': 'after',
            'isActualPointWriteExecute': 'false',
            'isMileageSubscriptionAlready': 'false',
            'isMileageSubscriptionReject': 'false',
            'page': p
        }
        return urllib.parse.urlencode(params)
 
    def inner_html(s, sl=0):
        ret = ''
        for idx in s.contents[sl:]:
            ret += idx.strip() if idx is str else str(idx)
        return ret
 
    def f_text(s):
        return inner_html(s[0]).strip() if len(s) else ''

    page = 1
    ret_list = []
    col_set = set()

    while 1:
        try:
            f = urllib.request.urlopen(
                "http://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?" + make_args(code, page))
            data = f.read().decode('utf-8')
        except Exception as e:
            print(e)
            break

        soup = bs4.BeautifulSoup(re.sub("&#(?![0-9])", "", data), "html.parser")
        cs = soup.select(".score_result li")
        if not len(cs):
            break

        for link in cs:
            try:
                url = link.select('.score_reple em a')[0].get('onclick')
            except ValueError:
                raise ValueError("[-] %d :" % page, data)

            m = re.search('[0-9]+', url)
            url = m.group(0) if m else ''

            if url in col_set:
                return ret_list

            col_set.add(url)

            cat = f_text(link.select('.star_score em'))
            cont = f_text(link.select('.score_reple p'))
            cont = re.sub('<span [^>]+>.+?</span>', '', cont)
            ret_list.append((url, cat, cont))
        page += 1
    return ret_list
 
 
def fetch(idx_):
    out_name = cfg.query_path + '%d.sql' % idx_

    try:
        if os.stat(out_name).st_size > 0:
            return
    except Exception as e:
        print(e)
        pass

    rs = get_comments(idx_)

    if not len(rs):
        return

    with open(out_name, 'w', encoding='utf-8') as f:
        f.write('INSERT IGNORE INTO movie VALUES ')

        for idx, r in enumerate(rs):
            if idx:
                f.write(',\n')
            f.write("(%d,%s,%s,'%s')" % (idx_, r[0], r[1], r[2].replace("'", "''").replace("\\", "\\\\")))

        f.write(';\n')
        f.close()

    time.sleep(0.5)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=cfg.n_threads) as executor:
        # 영화 고유 ID 값의 범위를 몰라서 대략 아래처럼 잡았습니다.
        for i in tqdm(range(10000, 200000)):
            executor.submit(fetch, i)
