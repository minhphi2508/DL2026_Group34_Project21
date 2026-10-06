"""Fetch only the six external pretrained files needed by the selected recipe."""
import argparse,bz2,json,os,shutil,time,urllib.request,zipfile
from pathlib import Path
from restoration.assets import ROOT,sha,read,write,contained,verify_models,verify_source

def download(row,cache):
    path=cache/row['name'];partial=cache/(row['name']+'.part');expected=row['bytes']
    if path.exists():
        if path.stat().st_size==expected and sha(path)==row['sha256']:return path
        raise ValueError('Cached archive failed verification: '+str(path)+'. Rename it and rerun.')
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>expected:raise ValueError('Partial archive is larger than expected: '+str(partial))
    while offset<expected:
        end=min(expected-1,offset+64*1024*1024-1)
        request=urllib.request.Request(row['url'],headers={'User-Agent':'DL2026-Group34-Project21','Accept-Encoding':'identity','Range':f'bytes={offset}-{end}'})
        with urllib.request.urlopen(request,timeout=90) as response:
            full=response.status==200 and offset==0
            if not full and (response.status!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{end}/{expected}'):
                raise RuntimeError('Server did not honor resume range. Preserve the .part file and retry later.')
            remaining=expected if full else end-offset+1
            with partial.open('wb' if full else 'ab') as f:
                while remaining:
                    block=response.read(min(1024*1024,remaining))
                    if not block:raise RuntimeError('Download interrupted. Rerun to resume.')
                    f.write(block);offset+=len(block);remaining-=len(block)
            print(f"{row['name']}: {offset/expected:.0%}",flush=True)
    if partial.stat().st_size!=expected or sha(partial)!=row['sha256']:raise ValueError('Downloaded archive SHA256 mismatch: '+row['name'])
    partial.replace(path);return path

def install(local_cache=None):
    manifest=read(ROOT/'provenance/MODELS.json');cache=ROOT/'.cache/downloads';cache.mkdir(parents=True,exist_ok=True)
    bypath={row['path']:row for row in manifest['models']};v3=contained('models/v3/twohead_candidate_v3_epoch14.pt')
    if not v3.exists() or sha(v3)!=bypath['models/v3/twohead_candidate_v3_epoch14.pt']['sha256']:
        raise ValueError('Tracked V3 checkpoint is missing or changed. Clone the complete repository again.')
    for archive in manifest['downloads']:
        needed=[bypath[path] for path in archive['models'] if not contained(path).is_file() or sha(contained(path))!=bypath[path]['sha256']]
        if not needed:continue
        existing=Path(local_cache)/archive['name'] if local_cache else None
        if existing and existing.is_file():
            if existing.stat().st_size!=archive['bytes'] or sha(existing)!=archive['sha256']:raise ValueError('Local archive SHA256 mismatch')
            path=existing
        else:path=download(archive,cache)
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as z:
                for row in needed:
                    target=contained(row['path']);target.parent.mkdir(parents=True,exist_ok=True)
                    part=target.with_name(target.name+'.installing')
                    # Explicit manifest members: no archive-directed extraction paths.
                    with z.open(row['archive_member']) as source,part.open('wb') as dest:shutil.copyfileobj(source,dest)
                    if part.stat().st_size!=row['bytes'] or sha(part)!=row['sha256']:raise ValueError('Extracted checkpoint mismatch: '+row['path'])
                    part.replace(target)
        else:
            row=needed[0];target=contained(row['path']);target.parent.mkdir(parents=True,exist_ok=True);part=target.with_name(target.name+'.installing')
            with bz2.open(path,'rb') as source,part.open('wb') as dest:shutil.copyfileobj(source,dest)
            if part.stat().st_size!=row['bytes'] or sha(part)!=row['sha256']:raise ValueError('Landmark model mismatch')
            part.replace(target)
    checked=verify_models();sources=verify_source()
    write(ROOT/'.cache/MODEL_SETUP.json',dict(status='PASS',models=checked,verified_source_files=sources))
    print('Models verified. Ready to restore images.',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');p.add_argument('--local-cache',type=Path)
    a=p.parse_args()
    if a.verify_only:
        print(json.dumps(dict(status='PASS',models=len(verify_models()),source_files=verify_source())))
    else:install(a.local_cache)

if __name__=='__main__':main()
