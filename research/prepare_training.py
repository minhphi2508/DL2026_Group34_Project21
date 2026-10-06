"""Reconstruct the unchanged sealed training bundle in an isolated directory."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from restoration.assets import ROOT,sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists():raise FileExistsError('Choose a new training work directory')
    src=ROOT/'research_archive/training/v3';manifest=json.loads((src/'BUNDLE_MANIFEST.json').read_text())
    resolved={}
    for name,expected in manifest['files'].items():
        source=src/('sealed_notes/operator_readme.txt' if name=='README_VI.md' else name)
        if sha(source)!=expected:raise ValueError('Sealed training source mismatch: '+name)
        resolved[name]=source
    out.mkdir(parents=True)
    for name,source in resolved.items():
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    shutil.copy2(src/'BUNDLE_MANIFEST.json',out/'BUNDLE_MANIFEST.json')
    print('Verified original training bundle: '+str(out))
if __name__=='__main__':main()
