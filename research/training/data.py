"""Pinned train/validation access; test pixels cannot be opened by this candidate."""
from __future__ import annotations
import csv, hashlib, importlib.util, json
from collections import Counter
from functools import lru_cache
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
MEAN=np.array([.485,.456,.406],np.float32)
STD=np.array([.229,.224,.225],np.float32)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

@lru_cache(maxsize=1)
def protocol():
    return json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))

def relative(name):
    p=PurePosixPath(str(name).replace('\\','/'))
    if p.is_absolute() or '..' in p.parts or ':' in str(p):raise ValueError('Unsafe data path')
    return str(p)

class FrozenData:
    def __init__(self, root):
        self.root=Path(root).resolve(); self.p=protocol()
        if not self.root.is_dir():raise ValueError('Extracted data directory required, not a ZIP')
        for name,key in [('dataset_v1/metadata/core_manifest.csv','core_manifest_sha256'),
                         ('benchmark_v1_candidate2/metadata/benchmark_manifest.csv','benchmark_manifest_sha256')]:
            if digest(self.root/name)!=self.p[key]:raise ValueError('Frozen metadata hash mismatch: '+name)
        core=self.csv('dataset_v1/metadata/core_manifest.csv')
        self.train=sorted([r for r in core if r['split']=='train' and r['decision']=='core'],key=lambda r:r['dataset_path'])
        if len(self.train)!=600 or len({r['dataset_path'] for r in self.train})!=600:raise ValueError('Expected600train')
        if any(not relative(r['dataset_path']).startswith('dataset_v1/core_old/train/') for r in self.train):raise ValueError('Nontraining source')
        self.val=sorted([r for r in self.csv('benchmark_v1_candidate2/metadata/benchmark_manifest.csv') if r['split']=='val'],key=lambda r:(r['suite'],r['severity'],r['id']))
        counts=Counter((r['suite'],r['severity']) for r in self.val)
        if len(self.val)!=1500 or len(counts)!=15 or set(counts.values())!={100}:raise ValueError('Incomplete all-track validation')
        if len({r['id'] for r in self.val})!=100:raise ValueError('Expected100valsourcephotos')
        gt={r['id']:r['gt_path'] for r in self.val}
        self.val += [dict(id=i,split='val',suite='clean',severity='none',input_path=p,gt_path=p,scratch_mask_path='',missing_mask_path='') for i,p in sorted(gt.items())]
        self.hashes=json.loads((HERE/'assets/validation_assets_sha256.json').read_text(encoding='utf-8'))
        if len(self.hashes)!=2800 or any('/val/' not in relative(p) for p in self.hashes):raise ValueError('Invalid pinned validation assets')
        self.checked=set(); self.opens=Counter()
        spec=importlib.util.spec_from_file_location('twohead_frozen_degradation',HERE/'assets/degradation_v1.py')
        self.degradation=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.degradation)
        import yaml
        self.cfg=yaml.safe_load((HERE/'assets/degradation_config_v1.yaml').read_text(encoding='utf-8'))

    def csv(self,name):
        return list(csv.DictReader(StringIO((self.root/relative(name)).read_text(encoding='utf-8-sig'))))

    def open_image(self,name,mode='RGB',training=False,expected=None):
        name=relative(name)
        prefix='dataset_v1/core_old/train/' if training else 'benchmark_v1_candidate2/val/'
        if not name.startswith(prefix):raise ValueError('Refusing pixel access outside train/val')
        if not training and name not in self.hashes:raise ValueError('Validation asset not pinned')
        path=(self.root/name).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Path escapes data root')
        if name not in self.checked:
            if digest(path)!=(expected if training else self.hashes[name]):raise ValueError('Payload hash mismatch: '+name)
            self.checked.add(name)
        self.opens['train' if training else 'val']+=1
        with Image.open(path) as im:return im.convert(mode).copy()

    def validation_sample(self,row):
        if row['split']!='val':raise ValueError('Only validation evaluation permitted')
        rgb=np.array(self.open_image(row['input_path']))
        masks=[]
        for field in ('scratch_mask_path','missing_mask_path'):
            masks.append(np.array(self.open_image(row[field],'L'))>0 if row[field] else np.zeros(rgb.shape[:2],bool))
        if any(m.shape!=rgb.shape[:2] for m in masks):raise ValueError('Mask extent mismatch')
        return rgb,masks[0],masks[1]

    def ordinary_schedule(self,epoch):
        rng=np.random.default_rng(np.random.SeedSequence([self.p['seed'],int(epoch)]))
        suites=[s for s,n in self.p['training']['mixture_counts'].items() for _ in range(n)]
        if len(suites)!=600:raise ValueError('Mixture must be exact600')
        rng.shuffle(suites); order=rng.permutation(600); counters={s:0 for s in suites}
        out=[]
        for idx,s in zip(order,suites):
            severity=('low','medium','high')[counters[s]%3];counters[s]+=1
            out.append((int(idx),s,severity))
        return out

    def schedule(self,epoch):
        from paired_sampling import draft_schedule
        if not isinstance(epoch,int) or epoch<0:raise ValueError('Nonnegative new-run epoch required')
        generation_epoch=self.p['training']['generation_epoch_offset']+epoch
        return draft_schedule(self,generation_epoch)

    def training_sample(self,item,epoch):
        from paired_sampling import render_sample
        expected=self.p['training']['generation_epoch_offset']+int(epoch)
        if not isinstance(item,dict) or item.get('generation_epoch')!=expected:raise ValueError('Sampling descriptor belongs to another generation epoch')
        rgb,target,_,meta=render_sample(self,item,epoch)
        return rgb,target,meta

def normalized(rgb):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:raise ValueError('Expected nonemptyRGBuint8')
    return np.ascontiguousarray(((rgb.astype(np.float32)/255.-MEAN)/STD).transpose(2,0,1))

def padded(rgb):
    normalized(rgb);h,w=rgb.shape[:2]
    return np.pad(rgb,((0,(-h)%32),(0,(-w)%32),(0,0)),mode='reflect' if min(h,w)>1 else 'edge'),(h,w)

def verify_bundle():
    manifest=json.loads((HERE/'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        path=(HERE/relative(name)).resolve()
        if not path.is_relative_to(HERE) or digest(path)!=expected:raise ValueError('Bundle hash mismatch: '+name)
    return digest(HERE/'protocol.json')
