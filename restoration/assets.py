"""Pinned assets and source verification shared by installation and inference."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def contained(relative):
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Asset path escapes repository')
    return path
def verify_models():
    manifest=read(ROOT/'provenance/MODELS.json');checked=[]
    for row in manifest['models']:
        path=contained(row['path'])
        if not path.is_file():raise FileNotFoundError(f"Missing model: {row['path']}. Run setup_models.py first.")
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Model integrity mismatch: '+row['path'])
        checked.append(dict(path=row['path'],sha256=row['sha256']))
    return checked
def verify_source():
    source=read(ROOT/'provenance/UPSTREAM_SOURCE.json')
    for row in source['files']:
        if sha(contained(row['path']))!=row['sha256']:raise ValueError('Pinned Wan source changed: '+row['path'])
    return len(source['files'])
