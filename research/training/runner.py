"""Explicit preflight/smoke/train/evaluate actions, isolated output and no test mode."""
from __future__ import annotations
import argparse, collections, hashlib, importlib.metadata, json, os, platform, re, time
from pathlib import Path
from data import HERE,FrozenData,digest,protocol,verify_bundle

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)

def output(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}',name):raise ValueError('Run name must be a simple label')
    path=(HERE/'runs'/name).resolve()
    if not path.is_relative_to(HERE/'runs'):raise ValueError('Output outside experiment')
    return path

def environment():
    from runtime_precision import configure
    import torch,torchvision,segmentation_models_pytorch as smp
    import numpy,PIL,yaml,timm,huggingface_hub,safetensors,tqdm
    precision_policy=configure()
    if not str(torch.__version__).startswith('2.8.0') or not str(torchvision.__version__).startswith('0.23.0') or smp.__version__!='0.5.0':raise RuntimeError('Use validatedtorch2.8/torchvision0.23/SMP0.5environment')
    # Match importlib's first-discovered distribution, rather than letting a
    # later shadowed site-packages overwrite the active environment's version.
    packages={};seen=collections.defaultdict(list)
    for distribution in importlib.metadata.distributions():
        name=distribution.metadata.get('Name')
        if not name:continue
        key=re.sub(r'[-_.]+','-',name).lower()
        packages.setdefault(key,distribution.version);seen[key].append(distribution.version)
    active={'numpy':numpy.__version__,'pillow':PIL.__version__,'pyyaml':yaml.__version__,
            'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
            'segmentation-models-pytorch':smp.__version__,'timm':timm.__version__,
            'huggingface-hub':huggingface_hub.__version__,'safetensors':safetensors.__version__,'tqdm':tqdm.__version__}
    if any(packages.get(k)!=v for k,v in active.items()):raise RuntimeError('Loaded module differs from active package metadata')
    package_hash=hashlib.sha256(json.dumps(packages,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),smp=smp.__version__,cuda_available=torch.cuda.is_available(),cuda_build=torch.version.cuda,
        package_versions=packages,package_versions_sha256=package_hash,active_module_versions=active,
        shadowed_distribution_versions={k:v for k,v in sorted(seen.items()) if len(v)>1},fp32_precision_policy=precision_policy)

def preflight(data,out):
    p=protocol();counts=dict(collections.Counter(item['suite'] for item in data.schedule(0)))
    if counts!=p['training']['mixture_counts']:raise ValueError('Wrong sampling mixture')
    for r in data.train:
        data.open_image(r['dataset_path'],training=True,expected=r['dataset_sha256'])
    for name in sorted(data.hashes):data.open_image(name,'L' if '/mask' in name else 'RGB')
    report=dict(status='PASS',protocol_sha256=verify_bundle(),train_photos=600,validation_conditions=1600,
                validation_asset_hashes=2800,training_asset_hashes=600,mixture=counts,test_payloads_opened=0,
                environment=environment(),training_executed=False)
    write_json(out/'preflight.json',report);print(json.dumps(report,indent=2))

def save_torch(path,value):
    import torch
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)

def train(data,out,device_name,resume=False):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,Dataset
    from model import FORMAT,initialize,normalized,TwoHeadLoss
    from evaluate import run_validation,save_validation
    env=environment();p=protocol();device=torch.device(device_name)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Full training requires explicit CUDA; use smoke for CPU')
    if torch.version.cuda!='12.8':raise RuntimeError('Validated Blackwell build requires CUDA12.8 torch wheel')
    marker=out/'contract_smoke_cuda.json'
    if not marker.is_file():raise RuntimeError('Run full-batch CUDA contract smoke first')
    proof=json.loads(marker.read_text(encoding='utf-8'))
    if proof['status']!='PASS' or proof['protocol_sha256']!=verify_bundle() or proof['bundle_sha256']!=digest(HERE/'BUNDLE_MANIFEST.json') or proof['batch_size']!=4 or proof['environment']!=env or proof['gpu_name']!=torch.cuda.get_device_name(device):raise RuntimeError('CUDA smoke not valid for this protocol/runtime')
    # Full payload verification must finish before any optimizer step.
    preflight(data,out)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(p['training']['threads']);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    model,transfer=initialize();model.to(device)
    loss_fn=TwoHeadLoss();optimizer=torch.optim.AdamW(model.parameters(),lr=p['training']['learning_rate'],weight_decay=p['training']['weight_decay'])
    sch=p['training']['scheduler'];scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=sch['factor'],patience=sch['patience'],min_lr=sch['min_lr'])
    protocol_hash=verify_bundle();bundle_hash=digest(HERE/'BUNDLE_MANIFEST.json');best=-1.;eligible_best=-1.;start_epoch=0;stale=0;best_epoch=eligible_epoch=0;budget_used=0.;previous_seconds=0.
    last=out/'last_completed.pt';amp=torch.cuda.is_bf16_supported()
    if (out/'training_status.json').exists() and not resume:raise RuntimeError('Existing run; resume explicitly or use a new run label')
    if resume:
        saved=torch.load(last,map_location='cpu',weights_only=True)
        if saved['format']!=FORMAT or saved['protocol_sha256']!=protocol_hash or saved['bundle_sha256']!=bundle_hash or not saved['completed_epoch'] or saved['environment']!=env:raise ValueError('Invalid epoch-boundary resume or changed runtime')
        model.load_state_dict(saved['model_state'],strict=True);optimizer.load_state_dict(saved['optimizer_state']);scheduler.load_state_dict(saved['scheduler_state'])
        torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
        start_epoch=saved['epoch'];best=saved['best_score'];eligible_best=saved['best_eligible_score'];stale=saved['stale_epochs']
        best_epoch=saved['best_epoch'];eligible_epoch=saved['best_eligible_epoch'];budget_used=saved['training_seconds_completed'];previous_seconds=saved['last_epoch_seconds']
    class TrainingDataset(Dataset):
        def __init__(self,epoch):self.epoch=epoch;self.items=data.schedule(epoch)
        def __len__(self):return 600
        def __getitem__(self,i):
            rgb,target,_=data.training_sample(self.items[i],self.epoch)
            return torch.from_numpy(normalized(rgb)),torch.from_numpy(target)
    out.mkdir(parents=True,exist_ok=True);last_epoch=start_epoch;run_start=time.perf_counter();budget_stopped=False
    try:
        for epoch in range(start_epoch,p['training']['epochs_max']):
            if budget_used+(time.perf_counter()-run_start)+previous_seconds>p['training']['wall_budget_seconds']:
                budget_stopped=True;break
            verify_bundle();write_json(out/'training_status.json',dict(status='TRAINING',epoch=epoch+1,last_completed_epoch=last_epoch,environment=env,protocol_sha256=protocol_hash,test_payloads_opened=0))
            model.train();loader=DataLoader(TrainingDataset(epoch),batch_size=p['training']['batch_size'],shuffle=False,num_workers=0)
            values=[];start=time.perf_counter()
            for rgb,target in loader:
                rgb=rgb.to(device);target=target.to(device);optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                    logits=model(rgb);loss,_=loss_fn(logits,target)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; abort before optimizer')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),p['training']['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step();values.append(float(loss.detach()))
            training_phase_seconds=time.perf_counter()-start
            verify_bundle();rows,summary,pngs=run_validation(model,data,device)
            summary.update(epoch=epoch+1,protocol_sha256=protocol_hash,train_loss=float(np.mean(values)),epoch_seconds=time.perf_counter()-start)
            summary['training_phase_seconds']=training_phase_seconds
            previous_seconds=summary['epoch_seconds']
            score=summary['score'];is_best=score>best+1e-4;is_eligible=summary['numerically_eligible'] and score>eligible_best
            stale=0 if is_best else stale+1
            if is_best:best=score;best_epoch=epoch+1
            if is_eligible:eligible_best=score;eligible_epoch=epoch+1
            scheduler.step(score)
            epoch_dir=out/'epochs'/f'epoch_{epoch+1:03d}';save_validation(epoch_dir,rows,summary,pngs if is_best or is_eligible else None)
            del pngs
            if not all(torch.isfinite(t).all() for t in model.state_dict().values()):raise FloatingPointError('Nonfinite completed state')
            inference=dict(format=FORMAT,classes=p['classes'],model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                epoch=epoch+1,completed_epoch=True,smoke_only=False,selection_eligible=summary['numerically_eligible'],protocol_sha256=protocol_hash,bundle_sha256=bundle_hash,
                source_checkpoint_sha256=p['source_checkpoint_sha256'],validation=summary,transfer=transfer,environment=env,promotion='RESEARCH_ONLY_NO_PRODUCTION')
            if is_best:save_torch(out/'best_unconstrained.pt',inference)
            if is_eligible:save_torch(out/'best_eligible.pt',inference)
            resume_state=dict(inference,optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),
                rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),best_score=best,best_eligible_score=eligible_best,stale_epochs=stale,
                best_epoch=best_epoch,best_eligible_epoch=eligible_epoch,training_seconds_completed=budget_used+time.perf_counter()-run_start,last_epoch_seconds=previous_seconds)
            save_torch(last,resume_state);last_epoch=epoch+1
            selected_epoch=eligible_epoch or best_epoch
            write_json(out/'selection.json',dict(selected_checkpoint='best_eligible.pt' if eligible_epoch else 'best_unconstrained.pt',
                selected_epoch=selected_epoch,selected_masks=f'epochs/epoch_{selected_epoch:03d}/predicted_masks.zip',
                numeric_gate_passed=bool(eligible_epoch),promotion='RESEARCH_ONLY_REQUIRES_REVIEW',protocol_sha256=protocol_hash,bundle_sha256=bundle_hash))
            write_json(out/'training_status.json',dict(status='COMPLETED_EPOCH',epoch=last_epoch,protocol_sha256=protocol_hash,validation=summary,precision='bfloat16' if amp else 'float32',test_payloads_opened=0))
            print(json.dumps({k:summary[k] for k in ('epoch','score','numerically_eligible','train_loss','epoch_seconds')}),flush=True)
            if stale>=p['training']['early_stopping_patience']:break
        verify_bundle();write_json(out/'completion.json',dict(status='BUDGET_BOUNDARY_STOPPED' if budget_stopped else 'TRAINING_COMPLETED',last_completed_epoch=last_epoch,numerically_eligible_checkpoint=(out/'best_eligible.pt').is_file(),quality_acceptance='PENDING_HUMAN_AND_END_TO_END_REVIEW',protocol_sha256=protocol_hash,test_payloads_opened=0))
    except Exception as exc:
        write_json(out/'failure.json',dict(status='FAILED_NOT_ACCEPTED',error=repr(exc),last_completed_epoch=last_epoch,resume='last_completed.pt_only',test_payloads_opened=0));raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','smoke','train','evaluate'])
    parser.add_argument('--data-root',required=True);parser.add_argument('--run-name',default='candidate_v3_pair_replay')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    verify_bundle();data=FrozenData(args.data_root);out=output(args.run_name)
    if args.action=='preflight':preflight(data,out)
    elif args.action=='smoke':
        from smoke import run_smoke
        run_smoke(data,out,args.device)
    elif args.action=='train':train(data,out,args.device,args.resume)
    else:
        import torch
        from model import FORMAT,create_model
        from evaluate import run_validation,save_validation
        saved=torch.load(out/'best_eligible.pt',map_location='cpu',weights_only=True)
        if saved['format']!=FORMAT or not saved['selection_eligible'] or saved['protocol_sha256']!=verify_bundle() or saved['bundle_sha256']!=digest(HERE/'BUNDLE_MANIFEST.json'):raise ValueError('Not eligible for evaluation')
        model=create_model();model.load_state_dict(saved['model_state'],strict=True);device=torch.device(args.device);model.to(device)
        rows,summary,pngs=run_validation(model,data,device);save_validation(out/'selected_validation',rows,summary,pngs);verify_bundle()
    verify_bundle()

if __name__=='__main__':main()
