"""LIGHT chat return and FULL epoch-resume backup, including failure evidence."""
from __future__ import annotations
import argparse, hashlib, json, re, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
RESTORE_PREFIX='candidate_v3_pair_replay/'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def selection(run):
    path=run/'selection.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

def pack_outputs(run_name='candidate_v3_pair_replay'):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}',run_name):raise ValueError('Unsafe run name')
    run=(HERE/'runs'/run_name).resolve()
    if not run.is_relative_to(HERE/'runs'):raise ValueError('Run escapes experiment')
    run.mkdir(parents=True,exist_ok=True);chosen=selection(run)
    small=[p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.txt','.log')]
    common=[p for p in HERE.glob('*') if p.is_file() and p.suffix in ('.py','.json','.ps1','.md') and not p.name.startswith('RETURN_FILES_')]
    light_files={p:'evidence/'+str(p.relative_to(run)).replace('\\','/') for p in small}
    light_files.update({p:'protocol/'+p.name for p in common})
    light_files.update({p:'protocol/assets/'+p.name for p in (HERE/'assets').glob('*') if p.is_file() and p.suffix!='.pt'})
    if chosen:
        for field,arc in [('selected_checkpoint','SELECTED_RESEARCH_CHECKPOINT.pt'),('selected_masks','SELECTED_PREDICTED_MASKS.zip')]:
            target=(run/chosen[field]).resolve()
            if not target.is_relative_to(run):raise ValueError('Selection escapes run')
            if not target.is_file():raise FileNotFoundError('Selected artifact missing: '+str(target))
            light_files[target]=arc
    light=HERE/('RETURN_LIGHT_'+run_name+'.zip');full=HERE/('BACKUP_FULL_RESUME_'+run_name+'.zip')
    def archive(path,files):
        temp=path.with_suffix('.zip.tmp')
        hashes={}
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
            for src,arc in sorted(files.items(),key=lambda item:item[1]):z.write(src,arc);hashes[arc]=sha(src)
            z.writestr('OUTPUT_SHA256.json',json.dumps(hashes,indent=2))
        temp.replace(path)
    archive(light,light_files)
    # Only the chosen epoch mask ZIP is needed for resume review; redundant masks
    # from earlier epochs remain on disk, not in FULL. All checkpoints/RNG included.
    full_files={p:RESTORE_PREFIX+'runs/'+run_name+'/'+str(p.relative_to(run)).replace('\\','/') for p in run.rglob('*') if p.is_file() and p.suffix!='.zip' and not p.name.endswith('.tmp')}
    if chosen:
        p=run/chosen['selected_masks'];full_files[p]=RESTORE_PREFIX+'runs/'+run_name+'/'+chosen['selected_masks']
    for p in HERE.glob('*'):
        if p.is_file() and p.suffix in ('.py','.json','.ps1','.md') and not p.name.startswith('RETURN_FILES_'):full_files[p]=RESTORE_PREFIX+p.name
    for p in (HERE/'assets').glob('*'):
        if p.is_file():full_files[p]=RESTORE_PREFIX+'assets/'+p.name
    archive(full,full_files)
    report=dict(status='PACKED',light=dict(path=str(light),bytes=light.stat().st_size,sha256=sha(light)),
        full_resume_backup=dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full)),
        light_under_250MB=light.stat().st_size<=250*1024*1024,selected=chosen,
        backup_required_before_reboot=['Original input bundle (includes train/val data)',full.name,light.name],
        scope='LIGHT sent in chat; FULL save to persistent Drive before quannet reboot; neither means production acceptance')
    (HERE/('RETURN_FILES_'+run_name+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['light_under_250MB']:raise RuntimeError('LIGHT exceeds250MiB; do not silently send oversized chat return')
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-name',default='candidate_v3_pair_replay');pack_outputs(p.parse_args().run_name)
