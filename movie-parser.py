# Credited by 'bab2min'
# original link : http://bab2min.tistory.com/556


from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import urllib
import urllib.request
import urllib.parse
import bs4
import re
import os
import time


def get_comments(code):
    def make_args(code, page):
        params = {
            'code': code,
            'type': 'after',
            'isActualPointWriteExecute': 'false',
            'isMileageSubscriptionAlready': 'false',
            'isMileageSubscriptionReject': 'false',
            'page': page
        }
        return urllib.parse.urlencode(params)
 
    def inner_html(s, sl=0):
        ret = ''
        for i in s.contents[sl:]:
            if i is str:
                ret += i.strip()
            else:
                ret += str(i)
        return ret
 
    def f_text(s):
        if len(s):
            return inner_html(s[0]).strip()
        return ''
 
    retList = []
    colSet = set()
    # print("Processing: %d" % code)
    
    page = 1
    while 1:
        try:
            f = urllib.request.urlopen(
                "http://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?" + make_args(code, page))
            data = f.read().decode('utf-8')
        except:
            break
        soup = bs4.BeautifulSoup(re.sub("&#(?![0-9])", "", data), "html.parser")
        cs = soup.select(".score_result li")
        if not len(cs):
            break

        for link in cs:
            try:
                url = link.select('.score_reple em a')[0].get('onclick')
            except:
                print(page)
                print(data)
                raise ""

            m = re.search('[0-9]+', url)
            if m:
                url = m.group(0)
            else:
                url = ''

            if url in colSet:
                return retList

            colSet.add(url)
            cat = f_text(link.select('.star_score em'))
            cont = f_text(link.select('.score_reple p'))
            cont = re.sub('<span [^>]+>.+?</span>', '', cont)
            retList.append((url, cat, cont))

        page += 1
 
    return retList
 
 
def fetch(idx):
    out_name = 'comments/%d.txt' % idx

    try:
        if os.stat(out_name).st_size > 0:
            return
    except:
        pass

    rs = get_comments(idx)

    if not len(rs):
        return

    with open(out_name, 'w', encoding='utf-8') as f:
        f.write('INSERT IGNORE INTO movie VALUES ')

        for idx, r in enumerate(rs):
            if idx:
                f.write(',\n')
            f.write("(%d,%s,%s,'%s')" % (i, r[0], r[1], r[2].replace("'", "''").replace("\\", "\\\\")))

        f.write(';\n')
        f.close()

    time.sleep(0.5)


# 영화 고유 ID값의 범위를 몰라서 대략 아래처럼 잡았습니다.
n_threads = 8
with ThreadPoolExecutor(max_workers=n_threads) as executor:
    for i in tqdm(range(10000, 200000)):
        executor.submit(fetch, i)
