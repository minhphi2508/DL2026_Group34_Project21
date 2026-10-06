"""Portable fixed-mask development probes; no training or threshold search."""
import argparse,csv,gc,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from restoration.assets import ROOT,read,write,sha,verify_models,verify_source
from restoration.pipeline import stage,choose_device

def rgb(path):
    with Image.open(path) as im:return np.array(im.convert('RGB'))
def gray(path):
    with Image.open(path) as im:return np.array(im.convert('L'))
def aligned(mask,size):return np.array(Image.fromarray(mask).resize(size,Image.Resampling.NEAREST))
def scores(image,reference):
    if image.shape!=reference.shape:raise ValueError('Reference geometry mismatch')
    mse=float(((image.astype(np.float64)-reference.astype(np.float64))**2).mean())
    return {'psnr':float(10*np.log10(255**2/max(mse,1e-12))),
        'ssim':float(structural_similarity(image,reference,channel_axis=2,data_range=255)),'mse':mse}
def segmentation(pred,target):
    pred=pred>0;target=target>0
    tp=int((pred&target).sum());fp=int((pred&~target).sum());fn=int((~pred&target).sum())
    return {'recall':tp/max(1,tp+fn),'precision':tp/max(1,tp+fp),'dice':2*tp/max(1,2*tp+fp+fn)}
def heads(detector,image):
    detector.precision();h,w=image.shape[:2]
    padded=np.pad(image,((0,(-h)%32),(0,(-w)%32),(0,0)),mode='reflect' if min(h,w)>1 else 'edge')
    mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32)
    normalized=((padded.astype(np.float32)/255.-mean)/std).transpose(2,0,1)
    tensor=detector.torch.from_numpy(np.ascontiguousarray(normalized)).unsqueeze(0).to(detector.device)
    with detector.torch.inference_mode():p=detector.model(tensor).sigmoid()[0,:,:h,:w].float().cpu().numpy()
    if not np.isfinite(p).all():raise ValueError('Nonfinite detector output')
    return [(p[0]>=.4).astype(np.uint8)*255,(p[1]>=.5).astype(np.uint8)*255]

def global_restore(out,inputs,masks,device):
    (out/'receipts').mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    gpu='0' if device=='cuda' else '-1'
    stage(out,'global',['--Scratch_and_Quality_restore','--test_input',inputs,'--test_mask',masks,
        '--outputs_dir',out/'global','--gpu_ids',gpu],device)
    return out/'global/restored_image'

def finish_faces(out,global_images,device):
    gpu='0' if device=='cuda' else '-1'
    (out/'receipts').mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    stage(out,'detect_faces',['--url',global_images,'--save_url',out/'faces'],device)
    face_files=list((out/'faces').glob('*.png'));final=out/'final';final.mkdir()
    for path in global_images.glob('*.png'):shutil.copy2(path,final/path.name)
    if face_files:
        stage(out,'face',['--old_face_folder',out/'faces','--old_face_label_folder',out/'empty_labels',
            '--name','Setting_9_epoch_100','--gpu_ids',gpu,'--load_size','256','--label_nc','18',
            '--no_instance','--preprocess_mode','resize','--batchSize','1',
            '--results_dir',out/'face_output','--no_parsing_map'],device)
        if len(list((out/'face_output/each_img').glob('*.png')))!=len(face_files):raise RuntimeError('Incomplete face output')
        stage(out,'blend',['--origin_url',global_images,'--replace_url',out/'face_output/each_img','--save_url',final],device)
    return final

def fixtures(study,case_ids):
    rows=[r for r in read(ROOT/'research/FIXTURES.json')['cases'] if r['study']==study]
    if case_ids:
        selected=set(case_ids)
        if not selected.issubset({r['id'] for r in rows}):raise ValueError('Unknown case identifier')
        rows=[r for r in rows if r['id'] in selected]
    for r in rows:
        for asset in r['assets'].values():
            path=(ROOT/asset['path']).resolve()
            if not path.is_relative_to(ROOT) or sha(path)!=asset['sha256']:raise ValueError('Fixture hash mismatch')
    return rows

def mask_sources(rows,out,device,masks_only):
    from restoration.detector import MissingDetector
    detector=MissingDetector(device);metrics=[];variants=['A_final','B_group','C_union']
    for variant in variants:
        for folder in ['input','mask']:(out/variant/folder).mkdir(parents=True)
    # Exact scratch detector inputs/outputs from the original study are supplied
    # as hash-pinned fixtures. Only the mask source varies in this experiment.
    for r in rows:
        paths={k:ROOT/v['path'] for k,v in r['assets'].items()};image=rgb(paths['processed_input']);h,w=image.shape[:2]
        own_scratch,missing=heads(detector,rgb(paths['native_input']))
        own_scratch=aligned(own_scratch,(w,h))
        missing=aligned(cv2.dilate(missing,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))),(w,h))
        ms_scratch=gray(paths['microsoft_scratch'])
        masks=[np.maximum(ms_scratch,missing),np.maximum(own_scratch,missing),np.maximum(np.maximum(ms_scratch,own_scratch),missing)]
        # Reference labels are never supplied to the detector or restorer.
        target=aligned(gray(paths['scratch_target']),(w,h)) if 'scratch_target' in paths else None
        for variant,mask in zip(variants,masks):
            name=r['id']+'.png';shutil.copy2(paths['processed_input'],out/variant/'input'/name)
            Image.fromarray(mask).save(out/variant/'mask'/name)
            result={'id':r['id'],'variant':variant,'mask_pixels':int((mask>0).sum()),
                'added_vs_final':int(((mask>0)&(masks[0]==0)).sum())}
            if target is not None:result.update(segmentation(mask,target))
            metrics.append(result)
    del detector;gc.collect()
    if not masks_only:
        for variant in variants:
            branch=out/variant;glob=global_restore(branch,branch/'input',branch/'mask',device)
            for r in rows:
                if 'reference' not in r['assets']:continue
                image=rgb(glob/(r['id']+'.png'));ref=rgb(ROOT/r['assets']['reference']['path'])
                if ref.shape!=image.shape:ref=np.array(Image.fromarray(ref).resize((image.shape[1],image.shape[0]),Image.Resampling.LANCZOS))
                record=next(m for m in metrics if m['id']==r['id'] and m['variant']==variant)
                record.update(scores(image,ref))
    return metrics

def denoising_order(rows,out,device):
    from research.denoising.inference import Denoiser
    weights=ROOT/'research/denoising/weights'
    if sha(weights/'ffdnet_gray_clip.pth')!='0b254d45dafc1ed04729b2206e0c09e5cc1e477e1d094e5a876ea45a34e5d84c':raise ValueError('FFDNet weight mismatch')
    denoise=Denoiser('ffdnet_gray',device=device,weights_dir=weights,blend=.75,sigma_scale=1.0,tile_size=0,threads=2)
    for variant in ['baseline','denoise_before','denoise_after','ffdnet_alone']:
        for folder in ['input','mask']:(out/variant/folder).mkdir(parents=True)
    for r in rows:
        inp=ROOT/r['assets']['input']['path'];mask=ROOT/r['assets']['mask']['path'];name=r['id']+'.png'
        raw=rgb(inp);filtered=denoise(raw)
        for variant in ['baseline','denoise_before']:
            Image.fromarray(filtered if variant=='denoise_before' else raw).save(out/variant/'input'/name)
            shutil.copy2(mask,out/variant/'mask'/name)
        Image.fromarray(filtered).save(out/'ffdnet_alone/input'/name)
    baseline=global_restore(out/'baseline',out/'baseline/input',out/'baseline/mask',device)
    before=global_restore(out/'denoise_before',out/'denoise_before/input',out/'denoise_before/mask',device)
    after=out/'denoise_after/global/restored_image';after.mkdir(parents=True)
    for path in baseline.glob('*.png'):Image.fromarray(denoise(rgb(path))).save(after/path.name)
    metrics=[]
    for variant,folder in [('baseline',baseline),('denoise_before',before),('denoise_after',after),('ffdnet_alone',out/'ffdnet_alone/input')]:
        for r in rows:
            ref=rgb(ROOT/r['assets']['reference']['path'])
            metrics.append({'id':r['id'],'variant':variant,'stage':'global_before_face',**scores(rgb(folder/(r['id']+'.png')),ref)})
    return metrics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',choices=['mask_sources','denoising_order'],required=True)
    p.add_argument('--output',type=Path);p.add_argument('--case',action='append');p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--masks-only',action='store_true');a=p.parse_args()
    rows=fixtures(a.study,a.case)
    print('Verified '+str(len(rows))+' fixed auxiliary cases.',flush=True)
    if a.verify_only:return
    if a.output is None:p.error('--output is required')
    if a.masks_only and a.study!='mask_sources':p.error('--masks-only applies to mask_sources')
    out=a.output.resolve()
    if out.exists():raise FileExistsError('Choose a new experiment output directory')
    device=choose_device(a.device);verify_models();verify_source();cv2.setNumThreads(1)
    out.mkdir(parents=True)
    write(out/'PROTOCOL.json',{'study':a.study,'cases':[r['id'] for r in rows],
        'device':device,'default_changed':False,'threshold_search':False,'independent_test':False,'source':sha(Path(__file__))})
    try:
        result=mask_sources(rows,out,device,a.masks_only) if a.study=='mask_sources' else denoising_order(rows,out,device)
        write(out/'RESULTS.json',result)
        fields=sorted({k for row in result for k in row})
        with (out/'RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(result)
        write(out/'STATUS.json',{'status':'COMPLETED','rows':len(result),'masks_only':a.masks_only})
        print(json.dumps(result,indent=2),flush=True)
    except Exception as exc:
        write(out/'STATUS.json',{'status':'FAILED','error':str(exc)});raise
if __name__=='__main__':main()
