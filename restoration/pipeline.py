"""Fixed S0 recipe. Stages run in isolated processes to avoid module-name clashes."""
import argparse,gc,json,os,shutil,subprocess,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np
from PIL import Image,UnidentifiedImageError
from .assets import ROOT,read,write,sha,verify_models,verify_source

STAGES=ROOT/'restoration/wan_stage.py'
def choose_device(requested):
    import torch
    device='cuda' if requested=='auto' and torch.cuda.is_available() else ('cpu' if requested=='auto' else requested)
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Use --device cpu or SETUP_RTX.cmd with a working NVIDIA driver.')
        # Exercise a real kernel, not just driver enumeration.
        x=torch.ones((2,2),device='cuda');(x@x).cpu();torch.cuda.synchronize()
    return device
def collect(source):
    if not source.exists():raise FileNotFoundError(source)
    single=source.is_file();files=[source] if single else sorted(p for p in source.iterdir() if p.is_file() and not p.name.startswith('.'))
    accepted=[];skipped=[];failures=[]
    for p in files:
        try:
            with Image.open(p) as im:
                im.load();w,h=im.size;fmt=im.format
                if min(w,h)<16:raise ValueError('Image is too small: minimum side must be 16 pixels')
                accepted.append(dict(source=str(p),source_sha256=sha(p),native_size=[w,h],format=fmt,
                    source_mode=im.mode,frames=getattr(im,'n_frames',1),selected_frame=0))
        except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError) as error:
            row=dict(source=str(p),error=str(error))
            if single or p.suffix.lower() in set(Image.registered_extensions())|{'.heic','.heif','.raw','.dng'}:failures.append(row)
            else:skipped.append(row)
    return accepted,skipped,failures
def stage(run,name,args,device):
    env=os.environ.copy();env['ASTRA_WAN_STAGE_RECEIPT']=str(run/'receipts'/(name+'.json'))
    env['ASTRA_WAN_DEVICE']=device
    log=run/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
    print('Running '+name+' ...',flush=True)
    with log.open('w',encoding='utf-8') as f:
        p=subprocess.run([sys.executable,str(STAGES),name,*map(str,args)],env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('Stage failed: '+name+'. See '+str(log))
def execute(source,output,device):
    started=time.perf_counter();rows,skipped,failures=collect(source)
    if not rows:raise ValueError('No decodable images. Use JPG/JPEG/JFIF/PNG/WebP or another installed Pillow codec.')
    if output.exists():raise FileExistsError('Choose a new output folder; existing outputs are preserved: '+str(output))
    if source.is_dir() and output.is_relative_to(source):raise ValueError('Output folder must be outside input folder')
    model_bindings=verify_models();source_count=verify_source();output.mkdir(parents=True)
    for folder in ['input_wan','masks','receipts','logs','restored','native_display','diagnostics']:(output/folder).mkdir()
    journal=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),pipeline='full_wan_v3_missing_S0',
        recipe=read(ROOT/'provenance/SELECTED_PIPELINE.json')['pipeline'],requested_input=str(source),device=device,
        model_bindings=model_bindings,verified_source_files=source_count,cases=rows,skipped=skipped,input_failures=failures,
        normalization='First frame RGB; no EXIF rotation or alpha preservation; original files unchanged',
        training=False,manual_masks=False,quality_score_available=False)
    write(output/'RUN.json',journal)
    try:
        for i,row in enumerate(rows,1):
            row['id']=f'image_{i:04d}';name=row['id']+'.png'
            with Image.open(row['source']) as im:
                rgb=im.convert('RGB');scale=min(1,512/max(rgb.size));small=rgb.resize(tuple(round(d*scale) for d in rgb.size),Image.Resampling.LANCZOS)
                if min(small.size)<16:raise ValueError('Extreme aspect ratio is outside the validated Microsoft frame: '+row['source'])
                small.save(output/'input_wan'/name)
        gpu='0' if device=='cuda' else '-1'
        stage(output,'detection',['--test_path',output/'input_wan','--output_dir',output/'wan_masks','--input_size','full_size','--GPU',gpu],device)
        from .detector import MissingDetector
        detector=MissingDetector(device);cv2.setNumThreads(1)
        for row in rows:
            name=row['id']+'.png';t=time.perf_counter()
            with Image.open(row['source']) as im:native=np.array(im.convert('RGB'))
            missing=detector.predict(native)
            Image.fromarray(missing).save(output/'diagnostics'/(row['id']+'_v3_missing_native.png'))
            missing=cv2.dilate(missing,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
            with Image.open(output/'wan_masks/mask'/name) as im:scratch=np.array(im.convert('L'));size=im.size
            missing=np.array(Image.fromarray(missing).resize(size,Image.Resampling.NEAREST));union=np.maximum(scratch,missing)
            Image.fromarray(union).save(output/'masks'/name);Image.fromarray(missing).save(output/'diagnostics'/(row['id']+'_v3_missing_processed.png'))
            row.update(processing_size=list(size),v3_seconds=time.perf_counter()-t,
                wan_scratch_pixels=int((scratch>0).sum()),combined_mask_pixels=int((union>0).sum()),
                V3_missing_added_pixels=int(((missing>0)&(scratch==0)).sum()),mask_sha256=sha(output/'masks'/name))
            print('Mask ready: '+Path(row['source']).name,flush=True)
        del detector;gc.collect()
        if device=='cuda':
            import torch
            torch.cuda.empty_cache()
        stage(output,'global',['--Scratch_and_Quality_restore','--test_input',output/'wan_masks/input','--test_mask',output/'masks','--outputs_dir',output/'global','--gpu_ids',gpu],device)
        stage(output,'detect_faces',['--url',output/'global/restored_image','--save_url',output/'faces'],device)
        faces=list((output/'faces').glob('*.png'))
        if faces:
            stage(output,'face',['--old_face_folder',output/'faces','--old_face_label_folder',output/'empty_labels','--name','Setting_9_epoch_100',
                '--gpu_ids',gpu,'--load_size','256','--label_nc','18','--no_instance','--preprocess_mode','resize','--batchSize','1',
                '--results_dir',output/'face_output','--no_parsing_map'],device)
            if len(list((output/'face_output/each_img').glob('*.png')))!=len(faces):raise RuntimeError('Incomplete face output')
        final=output/'final';final.mkdir()
        for row in rows:shutil.copy2(output/'global/restored_image'/(row['id']+'.png'),final/(row['id']+'.png'))
        if faces:stage(output,'blend',['--origin_url',output/'global/restored_image','--replace_url',output/'face_output/each_img','--save_url',final],device)
        for row in rows:
            name=row['id']+'.png'
            with Image.open(final/name) as im:
                im.verify()
            export_name=row['id']+'_'+Path(row['source']).stem[:80]+'.png'
            with Image.open(final/name) as im:
                if list(im.size)!=row['processing_size']:raise RuntimeError('Output geometry changed')
                im.resize(tuple(row['native_size']),Image.Resampling.LANCZOS).save(output/'native_display'/export_name)
            shutil.copy2(final/name,output/'restored'/export_name)
            row.update(output='restored/'+export_name,native_display='native_display/'+export_name,
                faces=[f.name for f in faces if f.name.startswith(row['id']+'_')],output_sha256=sha(output/'restored'/export_name))
            if sha(row['source'])!=row['source_sha256']:raise RuntimeError('Input changed during inference')
        journal.update(status='COMPLETED_WITH_INPUT_FAILURES' if failures else 'COMPLETED',seconds=time.perf_counter()-started,
            restored_images=len(rows),face_crops=len(faces),native_display_is_superresolution=False,
            caution='Inspect before/after; detected faces may include false positives, severe damage may remain, face detail may change.')
        write(output/'RUN.json',journal)
        print('Restored images: '+str(output/'restored'),flush=True)
        print('Original-size display copies: '+str(output/'native_display'),flush=True)
        return 1 if failures else 0
    except Exception as error:
        journal.update(status='FAILED',error=str(error),seconds=time.perf_counter()-started);write(output/'RUN.json',journal);raise
def main(argv=None):
    p=argparse.ArgumentParser(description='Restore photos using Microsoft Bringing Old Photos Back to Life + Final Version missing-region masks.')
    p.add_argument('--input',required=True,type=Path,help='One image or a folder; folder scan is non-recursive')
    p.add_argument('--output',type=Path,help='New output folder; omitted creates outputs/run_<timestamp>')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args(argv)
    try:
        source=a.input.resolve();out=a.output.resolve() if a.output else ROOT/'outputs'/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        return execute(source,out,choose_device(a.device))
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        print('ERROR: '+str(error),file=sys.stderr);return 2
