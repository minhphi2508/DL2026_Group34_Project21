"""Microsoft baseline versus final masks on user-supplied photographs."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from restoration.pipeline import execute,choose_device
from research.run_experiments import global_restore,finish_faces,gray
from restoration.assets import write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    out=a.output.resolve()
    if out.exists():raise FileExistsError('Choose a new comparison output directory')
    out.mkdir(parents=True);device=choose_device(a.device)
    code=execute(a.input.resolve(),out/'final_pipeline',device)
    baseline=out/'microsoft_baseline';baseline.mkdir()
    global_images=global_restore(baseline,out/'final_pipeline/wan_masks/input',out/'final_pipeline/wan_masks/mask',device)
    finish_faces(baseline,global_images,device)
    rows=json.loads((out/'final_pipeline/RUN.json').read_text(encoding='utf-8'))['cases']
    result=[]
    for row in rows:
        name=row['id']+'.png';ms=gray(out/'final_pipeline/wan_masks/mask'/name);final=gray(out/'final_pipeline/masks'/name)
        result.append({'id':row['id'],'source':row['source'],'added_mask_pixels':int(((final>0)&(ms==0)).sum()),
            'paired_quality_score':None})
    write(out/'COMPARISON.json',{'cases':result,'clean_reference_available':False,
        'before_face':'final_pipeline/global/restored_image','after_face':'final_pipeline/final',
        'baseline_final':'microsoft_baseline/final','exit_code':code})
    print('Comparison saved: '+str(out));return code
if __name__=='__main__':raise SystemExit(main())
