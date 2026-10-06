"""Create a Python 3.12 environment and install the CPU or CUDA 12.8 profile."""
import argparse,os,subprocess,sys,venv
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def call(*args):subprocess.run(list(map(str,args)),cwd=ROOT,check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);a=p.parse_args()
    if sys.version_info[:2]!=(3,12) or sys.maxsize<2**32:raise RuntimeError('Use Python 3.12 64-bit on Windows.')
    if os.name!='nt':raise RuntimeError('This installer is for Windows.')
    env=ROOT/'.venv';python=env/'Scripts/python.exe'
    if not python.exists():venv.EnvBuilder(with_pip=True).create(env)
    call(python,'-m','pip','install','--upgrade','pip==26.2.1')
    index='https://download.pytorch.org/whl/'+('cu128' if a.device=='cuda' else 'cpu')
    call(python,'-m','pip','install','--force-reinstall','--no-deps','torch==2.8.0','torchvision==0.23.0','--index-url',index)
    call(python,'-m','pip','install','-r','requirements.txt')
    call(python,'-m','pip','check')
    call(python,'doctor.py','--device',a.device,'--skip-models')
    call(python,'setup_models.py')
    call(python,'doctor.py','--device',a.device)
    print('Setup complete. Put photos in inputs, then run RUN_RESTORATION.cmd.',flush=True)
if __name__=='__main__':
    try:main()
    except (RuntimeError,subprocess.SubprocessError,OSError) as error:
        print('SETUP FAILED: '+str(error),file=sys.stderr);raise SystemExit(1)
