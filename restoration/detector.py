"""V3 missing head, preserving the audited native-resolution inference recipe."""
import numpy as np
from .assets import ROOT,read,sha,contained

class MissingDetector:
    def __init__(self,device):
        import torch
        import segmentation_models_pytorch as smp
        self.torch=torch;self.device=torch.device(device)
        descriptor=read(ROOT/'provenance/V3_DESCRIPTOR.json');assets={}
        for key in ['checkpoint','protocol','bundle_manifest']:
            d=descriptor[key];path=contained(d['path'])
            if sha(path)!=d['sha256']:raise ValueError('V3 identity mismatch: '+key)
            assets[key]=path
        protocol=read(assets['protocol'])
        if protocol['classes']!=['visible_scratch','missing'] or protocol['validation']['thresholds']!={'scratch':.4,'missing':.5}:
            raise ValueError('V3 channel/threshold semantics changed')
        self.precision()
        saved=torch.load(assets['checkpoint'],map_location='cpu',weights_only=True)
        checks=dict(format='astra_missing_twohead_research_v1',classes=protocol['classes'],smoke_only=False,
            completed_epoch=True,epoch=descriptor['epoch'],selection_eligible=descriptor['numeric_gate_passed'],
            protocol_sha256=descriptor['protocol']['sha256'],bundle_sha256=descriptor['bundle_manifest']['sha256'],
            source_checkpoint_sha256=protocol['source_checkpoint_sha256'])
        for key,value in checks.items():
            if saved.get(key)!=value:raise ValueError('V3 checkpoint metadata changed: '+key)
        self.model=smp.Unet(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=2,activation=None)
        expected=self.model.state_dict();state=saved['model_state']
        if set(state)!=set(expected):raise ValueError('V3 checkpoint keys changed')
        for key,value in state.items():
            if value.shape!=expected[key].shape or value.dtype!=expected[key].dtype or not torch.isfinite(value).all():
                raise ValueError('Invalid V3 checkpoint tensor: '+key)
        self.model.load_state_dict(state,strict=True);self.model.to(self.device).eval()
    def precision(self):
        torch=self.torch;torch.set_num_threads(2)
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    def predict(self,rgb):
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected RGB uint8 image')
        self.precision();h,w=rgb.shape[:2]
        padded=np.pad(rgb,((0,(-h)%32),(0,(-w)%32),(0,0)),mode='reflect' if min(h,w)>1 else 'edge')
        mean=np.array([.485,.456,.406],np.float32);std=np.array([.229,.224,.225],np.float32)
        normalized=((padded.astype(np.float32)/255.-mean)/std).transpose(2,0,1)
        tensor=self.torch.from_numpy(np.ascontiguousarray(normalized)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():probability=self.model(tensor).sigmoid()[0,:,:h,:w].float().cpu().numpy()
        if not np.isfinite(probability).all():raise ValueError('Nonfinite V3 output')
        return (probability[1]>=.5).astype(np.uint8)*255
