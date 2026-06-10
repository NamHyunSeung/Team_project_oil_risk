"""OilPrice.com sitemap 기반 17년 헤드라인 수집 (2010-01 ~ 2026-06)
- 월별 sitemap_articles_YYYY_M.xml에서 <loc>(URL)과 <lastmod>(날짜) 추출
- URL slug에서 제목 복원 (예: How-Low-Could-US-Oil-Production-Actually-Go -> How Low Could US Oil Production Actually Go)
출력: output/oilprice_headlines.csv (date, title, category, url)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import requests
import time
import pandas as pd
from xml.etree import ElementTree as ET

NS = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
HEADERS = {'User-Agent': 'Mozilla/5.0 (research data collection)'}


def slug_to_title(url):
    path = url.split('oilprice.com/', 1)[-1]
    parts = path.rsplit('.html', 1)[0].split('/')
    category = '/'.join(parts[:-1])
    title = parts[-1].replace('-', ' ')
    return title, category


rows = []
for year in range(2010, 2027):
    for month in range(1, 13):
        if year == 2026 and month > 6:
            break
        url = f'https://oilprice.com/sitemap_articles_{year}_{month}.xml'
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except Exception as e:
            print(f'{year}-{month:02d}: ERROR {e}')
            time.sleep(0.5)
            continue
        if r.status_code != 200:
            print(f'{year}-{month:02d}: status={r.status_code}')
            time.sleep(0.5)
            continue
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as e:
            print(f'{year}-{month:02d}: PARSE ERROR {e}')
            time.sleep(0.5)
            continue
        n = 0
        for url_el in root.findall('sm:url', NS):
            loc = url_el.find('sm:loc', NS).text
            lastmod_el = url_el.find('sm:lastmod', NS)
            lastmod = lastmod_el.text if lastmod_el is not None else None
            title, category = slug_to_title(loc)
            date = lastmod[:10] if lastmod else None
            rows.append({'date': date, 'title': title, 'category': category, 'url': loc})
            n += 1
        print(f'{year}-{month:02d}: {n}건')
        time.sleep(0.5)

df = pd.DataFrame(rows)
df.to_csv('output/oilprice_headlines.csv', index=False)
print(f'\n총 {len(df)}건, {df["date"].min()} ~ {df["date"].max()}')
print('\n카테고리 분포 (상위 20):')
print(df['category'].value_counts().head(20))
