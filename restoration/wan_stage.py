"""Run pinned upstream Wan stages with strict CPU-safe checkpoint loading.

No model topology or inference math is replaced. Source remains unmodified.
"""
import hashlib,json,os,runpy,sys,time
from pathlib import Path
import torch
import cv2
cv2.setNumThreads(1)

ROOT=Path(__file__).resolve().parents[1]
UP=ROOT/'third_party/wan'
stage=sys.argv.pop(1)
scripts={'detection':('Global','detection.py'),'global':('Global','test.py'),
         'detect_faces':('Face_Detection','detect_all_dlib.py'),
         'face':('Face_Enhancement','test_face.py'),
         'blend':('Face_Detection','align_warp_back_multiple_dlib.py')}
folder,script=scripts[stage];base=UP/folder
os.chdir(base);sys.path.insert(0,str(base));sys.argv[0]=str(base/script)
torch.set_num_threads(2);torch.manual_seed(20260930)
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
loaded=[]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def strict_load(net,path):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    state=torch.load(path,map_location='cpu',weights_only=True)
    expected=net.state_dict();ignored=[]
    # PyTorch historically serialized InstanceNorm running stats even when
    # track_running_stats=False. Upstream loader already discards these.
    for key in list(state):
        if key in expected:continue
        module_name,_,buffer=key.rpartition('.')
        module=dict(net.named_modules()).get(module_name)
        if isinstance(module,torch.nn.modules.instancenorm._InstanceNorm) and not module.track_running_stats and buffer in ('running_mean','running_var','num_batches_tracked'):
            ignored.append(key);del state[key]
        else:raise RuntimeError('Unexpected checkpoint key: '+key)
    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):raise RuntimeError('Non-finite checkpoint')
    net.load_state_dict(state,strict=True)
    loaded.append({'path':str(path.resolve()),'sha256':sha(path),'state_keys':len(state),'ignored_nontracking_instance_norm_buffers':ignored})
    print('STRICTLY_LOADED',path,flush=True)
if stage=='global':
    from models.base_model import BaseModel
    def load_network(self,network,network_label,epoch_label,save_dir=''):
        strict_load(network,Path(save_dir or self.save_dir)/f'{epoch_label}_net_{network_label}.pth')
    BaseModel.load_network=load_network
elif stage=='face':
    from util import util
    def load_network(net,label,epoch,opt):
        strict_load(net,Path(opt.checkpoints_dir)/opt.name/f'{epoch}_net_{label}.pth');return net
    util.load_network=load_network
elif stage=='detection':
    original_load=torch.load
    def load_cpu(*args,**kwargs):
        kwargs['map_location']='cpu';kwargs['weights_only']=True
        return original_load(*args,**kwargs)
    torch.load=load_cpu
elif stage=='blend':
    # skimage 0.26 preserves uint8 on nearest-neighbor warp. The original
    # pipeline expects a floating mask and multiplies it in-place by 255.0.
    # Restore the historical dtype; coordinates and pixel values are unchanged.
    import numpy as np
    from skimage import transform
    original_warp=transform.warp
    def warp_float_mask(*args,**kwargs):
        result=original_warp(*args,**kwargs)
        return result.astype(np.float64,copy=False) if kwargs.get('order')==0 else result
    transform.warp=warp_float_mask

started=time.perf_counter()
runpy.run_path(str(base/script),run_name='__main__')
receipt=os.environ.get('ASTRA_WAN_STAGE_RECEIPT')
if receipt:
    Path(receipt).write_text(json.dumps({'stage':stage,'status':'COMPLETED','seconds':time.perf_counter()-started,
        'cpu_threads':2,'torch':torch.__version__,'device':os.environ.get('ASTRA_WAN_DEVICE','cpu'),'checkpoints':loaded},indent=2),encoding='utf-8')

