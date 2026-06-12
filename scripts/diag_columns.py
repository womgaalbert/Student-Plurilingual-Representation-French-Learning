import re, sys
sys.path.insert(0, 'src')
import pandas as pd
from utils.constants import CSV_COLUMN_MAP

def _norm(s):
    s = s.lower().strip()
    s = re.sub(r'[^\x00-\x7f]', '', s)
    s = s.replace("'", '').replace('"', '').replace('`', '').replace('-', ' ')
    return re.sub(r'\s+', ' ', s)

df = pd.read_csv('data/raw/data_FLP.csv')
sorted_map = sorted(CSV_COLUMN_MAP.items(), key=lambda kv: len(kv[0]), reverse=True)
rename = {}
used_shorts = set()
for col in df.columns:
    col_n = _norm(col)
    for csv_name, short in sorted_map:
        if short in used_shorts:
            continue
        csv_n = _norm(csv_name)
        if not col_n or not csv_n:
            continue
        ratio = min(len(col_n), len(csv_n)) / max(len(col_n), len(csv_n))
        if ratio >= 0.50 and (csv_n in col_n or col_n in csv_n):
            rename[col] = short
            used_shorts.add(short)
            break

df2 = df.rename(columns=rename)
short_names = list(CSV_COLUMN_MAP.values())
present = [s for s in short_names if s in df2.columns]
missing = [s for s in short_names if s not in df2.columns]
print(f'Short names present: {len(present)}/{len(short_names)}')
print(f'Missing short names: {missing}')
print()
print('Full mapping:')
for col in df.columns:
    mapped = rename.get(col, '[UNMAPPED]')
    print(f'  {mapped:40s} <- {repr(col[:65])}')
