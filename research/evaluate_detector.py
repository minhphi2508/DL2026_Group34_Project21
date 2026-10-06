"""Evaluate the reviewed selected checkpoint without changing numerical gates."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from restoration.assets import ROOT,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    sys.path.insert(0,str(ROOT/'research_archive/training/v3'))
    from data import FrozenData
    data=FrozenData(a.data_root)
    # The same pinned hashes govern archived and selected-model evaluation.
    for r in data.train:
        if sha(data.root/r['dataset_path'])!=r['dataset_sha256']:raise ValueError('Training source mismatch: '+r['dataset_path'])
    for name,expected in data.hashes.items():
        if sha(data.root/name)!=expected:raise ValueError('Validation asset mismatch: '+name)
    print('Verified 600 training images and 2,800 validation assets; no test pixels opened.',flush=True)
    if a.verify_only:return
    if a.output is None:p.error('--output is required for evaluation')
    out=a.output.resolve()
    if out.exists():raise FileExistsError('Choose a new evaluation output directory')
    from restoration.pipeline import choose_device
    from restoration.detector import MissingDetector
    from evaluate import run_validation,save_validation
    detector=MissingDetector(choose_device(a.device))
    rows,summary,pngs=run_validation(detector.model,data,detector.device)
    save_validation(out,rows,summary,pngs)
    print(json.dumps({'scratch_dice':summary['groups']['scratch_positive']['dice'],
        'missing_dice':summary['groups']['missing_positive']['dice'],
        'missing_recall':summary['groups']['missing_positive']['recall'],
        'numerically_eligible':summary['numerically_eligible'],'conditions':len(rows),'output':str(out)},indent=2))
if __name__=='__main__':main()
