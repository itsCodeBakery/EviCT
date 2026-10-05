from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / 'vlmDiagnosis'
LOCK = json.loads((DX / 'CORE_LOCK.json').read_text())
MANIFEST = ROOT / LOCK['frozen_core']['manifest']

def sha256_file(path, chunk=8*1024*1024):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

if sha256_file(MANIFEST) != LOCK['frozen_core']['manifest_sha256']:
    print('CORE LOCK MANIFEST CHANGED')
    sys.exit(2)

missing = []
changed = []

with MANIFEST.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

for row in rows:
    p = ROOT / row['path']
    if not p.exists():
        missing.append(row['path'])
        continue
    if sha256_file(p) != row['sha256']:
        changed.append(row['path'])

if missing or changed:
    print('EViCT CORE LOCK VIOLATION')
    if missing:
        print('Missing:')
        for x in missing:
            print(' -', x)
    if changed:
        print('Changed:')
        for x in changed:
            print(' -', x)
    sys.exit(3)

print(f'EViCT CORE VERIFIED — {len(rows)} locked tracked files unchanged')
