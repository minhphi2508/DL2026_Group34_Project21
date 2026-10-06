"""Two-output candidate and loss; no production imports or pretrained downloads."""
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from data import HERE, digest, protocol, normalized, padded
from runtime_precision import configure

FORMAT='astra_missing_twohead_research_v1'

def create_model():
    configure()
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=2,activation=None)

def initialize():
    path=HERE/'assets/parent_v2_epoch20.pt'; p=protocol()
    if digest(path)!=p['source_checkpoint_sha256']:raise ValueError('Base checkpoint hash mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if (saved.get('format')!=FORMAT or saved.get('epoch')!=p['parent_epoch'] or
            saved.get('smoke_only') is not False or saved.get('completed_epoch') is not True or
            saved.get('classes')!=p['classes'] or saved.get('protocol_sha256')!=p['parent_protocol_sha256'] or
            saved.get('selection_eligible') is not False or
            saved.get('bundle_sha256')!=p['parent_bundle_sha256'] or
            saved.get('source_checkpoint_sha256')!=p['ancestor_frozen_checkpoint_sha256']):
        raise ValueError('Invalid pinned research parent')
    model=create_model(); state=model.state_dict();old=saved['model_state']
    if set(old)!=set(state):raise ValueError('Unexpected base checkpoint keys')
    transferred=[]
    for k,t in state.items():
        if old[k].shape!=t.shape or old[k].dtype!=t.dtype or not torch.isfinite(old[k]).all():raise ValueError('Unexpected or nonfinite parent tensor: '+k)
        t.copy_(old[k]);transferred.append(k)
    model.load_state_dict(state,strict=True)
    return model,dict(source_sha256=digest(path),source_epoch=p['parent_epoch'],
        all_state_keys_transferred=len(transferred),scratch_head_transferred=True,missing_head_transferred=True,
        parent_numeric_gate_passed=False,parent_protocol_sha256=p['parent_protocol_sha256'],
        parent_bundle_sha256=p['parent_bundle_sha256'],optimizer_state_transferred=False,
        scheduler_state_transferred=False,rng_state_transferred=False)

class TwoHeadLoss(nn.Module):
    def forward(self,logits,target):
        if logits.shape!=target.shape or logits.ndim!=4 or logits.shape[1]!=2:raise ValueError('Expected Bx2xHxW matching targets')
        if not torch.isfinite(target).all() or target.min()<0 or target.max()>1:raise ValueError('Invalid targets')
        logits=logits.float();target=target.float();cfg=protocol()['training']['loss']
        terms=[];info={}
        for channel,name in enumerate(('scratch','missing')):
            l=logits[:,channel];t=target[:,channel];prob=l.sigmoid().flatten(1);labels=t.flatten(1)
            bce=nn.functional.binary_cross_entropy_with_logits(l,t)
            sums=labels.sum(1);dice=1-(2*(prob*labels).sum(1)+cfg['smooth'])/(prob.sum(1)+sums+cfg['smooth'])
            dice=torch.where(sums>0,dice,prob.mean(1)).mean()
            term=cfg['bce_weight']*bce+cfg['dice_weight']*dice;terms.append(term*cfg['head_weights'][channel])
            info[name+'_bce']=float(bce.detach());info[name+'_dice_loss']=float(dice.detach())
        return sum(terms)/sum(cfg['head_weights']),info

def probability(model,rgb,device):
    configure()
    image,(h,w)=padded(rgb)
    tensor=torch.from_numpy(normalized(image)).unsqueeze(0).to(device)
    with torch.inference_mode():out=model(tensor).sigmoid()[0,:,:h,:w]
    array=out.float().cpu().numpy()
    if array.shape!=(2,h,w) or not np.isfinite(array).all():raise ValueError('Invalid probability output')
    return array

def masks(model,rgb,device):
    probs=probability(model,rgb,device);thresholds=protocol()['validation']['thresholds']
    scratch=probs[0]>=thresholds['scratch'];missing=probs[1]>=thresholds['missing']
    return {'scratch':scratch.astype(np.uint8)*255,'missing':missing.astype(np.uint8)*255,'union':(scratch|missing).astype(np.uint8)*255}
