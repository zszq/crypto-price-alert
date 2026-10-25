"""从下载站的 S3 存储桶列出某月全部逐笔成交文件的大小，存 trade_sizes_<月>.json。

pool.py 用它只留下载站有文件的合约，download.py 用它校验下载是否完整。

download.gatedata.org 会丢掉列表查询参数，只能直连存储桶。用法：list_sizes.py 2026MM ...
"""

import json
import re
import sys
import urllib.parse
import urllib.request

import sim

BUCKET = "https://gateio-public-data.s3.amazonaws.com/?list-type=2&prefix=futures_usdt/trades/{m}/"

for month in sys.argv[1:]:
    sizes, token = {}, None
    while True:
        url = BUCKET.format(m=month) + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        page = urllib.request.urlopen(url, timeout=60).read().decode()
        for name, size in re.findall(
            r"<Key>futures_usdt/trades/\d+/(\w+?)-\d+\.csv\.gz</Key>.*?<Size>(\d+)</Size>", page
        ):
            sizes[name] = int(size)
        # 每页最多 1000 条
        match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", page)
        if not match:
            break
        token = match.group(1)
    json.dump(sizes, open(sim.DATA / f"trade_sizes_{month}.json", "w"))
    print(month, len(sizes), "个文件")
