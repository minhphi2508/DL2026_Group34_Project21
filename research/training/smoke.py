"""Real CPU/CUDA forward/backward contract checks; no optimizer update/checkpoint."""
import gc, json, os, time
from collections import Counter
import numpy as np
import torch
from data import HERE,digest,normalized,padded,protocol,verify_bundle
from model import initialize,TwoHeadLoss,masks,probability
from runtime_precision import configure,snapshot

def run_smoke(data,out,device_name):
    configure()
    from runner import environment,write_json
    from evaluate import segmentation,summarize
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=protocol();env=environment();device=torch.device(device_name);torch.set_num_threads(p['training']['threads']);torch.manual_seed(p['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; CPU smoke cannot certify GPU')
    model,transfer=initialize();model.to(device).eval();checks={};start=time.perf_counter()
    # Both learned heads are inherited exactly; no reset to a missing prior.
    saved=torch.load(HERE/'assets/parent_v2_epoch20.pt',map_location='cpu',weights_only=True)
    import segmentation_models_pytorch as smp
    old=smp.Unet(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=2,activation=None).to(device).eval()
    old.load_state_dict(saved['model_state'],strict=True)
    probe=np.random.default_rng(p['seed']).integers(0,256,(137,219,3),dtype=np.uint8)
    padded_rgb,(h,w)=padded(probe)
    x=torch.from_numpy(normalized(padded_rgb)).unsqueeze(0).to(device)
    with torch.inference_mode():old_prob=old(x).sigmoid()[0,:,:h,:w].float().cpu().numpy()
    new_prob=probability(model,probe,device)
    delta=float(np.max(np.abs(old_prob-new_prob)))
    write_json(out/'initialization_parity.json',dict(
        status='PASS' if np.isfinite(delta) and delta<=2e-5 else 'FAIL',
        both_heads_max_probability_delta=delta,unchanged_tolerance=2e-5,fp32_precision_policy=snapshot(),
        parent_checkpoint_sha256=p['source_checkpoint_sha256'],parent_epoch=p['parent_epoch'],
        protocol_sha256=verify_bundle(),bundle_sha256=digest(HERE/'BUNDLE_MANIFEST.json'),
        optimizer_steps=0,test_payloads_opened=0))
    if not np.isfinite(old_prob).all():raise AssertionError('Frozen-reference output nonfinite')
    if delta>2e-5:raise AssertionError('Parent two-head output initialization differs')
    checks['initial_both_heads_max_probability_delta']=delta
    del old,saved;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    for hh,ww in ((1,1),(1,43),(43,1),(137,219)):
        image=np.zeros((hh,ww,3),np.uint8);mm=masks(model,image,device)
        assert all(v.shape==(hh,ww) and v.dtype==np.uint8 and set(np.unique(v))<={0,255} for v in mm.values())
        assert np.array_equal(mm['union']>0,(mm['scratch']>0)|(mm['missing']>0))
    checks['odd_and_singleton_mask_extent']='PASS'
    invalid=[np.zeros((3,3),np.uint8),np.zeros((3,3,3),np.float32),np.zeros((0,3,3),np.uint8),np.zeros((3,3,4),np.uint8)]
    for bad in invalid:
        try:normalized(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid RGB accepted')
    checks['invalid_rgb_rejected']=len(invalid)
    for bad in ('benchmark_v1_candidate2/test/noise/low/input/no.png','dataset_v1/core_old/test/HKM/no.jpg','../outside.png'):
        try:data.open_image(bad)
        except ValueError:pass
        else:raise AssertionError('Out-of-scope payload accepted')
    checks['test_or_path_escape_rejected']=3
    schedule=data.schedule(0);assert len(schedule)==600 and len({v['index'] for v in schedule})==510
    assert dict(Counter(v['suite'] for v in schedule))==p['training']['mixture_counts'];assert schedule==data.schedule(0)
    checks['exact_train_sampling_and_replay']='PASS'
    samples={}
    for suite in p['training']['mixture_counts']:
        item=next(v for v in schedule if v['suite']==suite)
        rgb,target,info=data.training_sample(item,0);rgb2,t2,_=data.training_sample(item,0)
        assert np.array_equal(rgb,rgb2) and np.array_equal(target,t2)
        assert target.shape==(2,512,512) and not (target[0].astype(bool)&target[1].astype(bool)).any()
        if suite in ('clean','noise','age_quality'):assert target.sum()==0
        if suite=='missing':assert target[0].sum()==0 and target[1].sum()>0
        if suite=='scratch':assert target[1].sum()==0 and target[0].sum()>0
        if suite=='combined':assert target[0].sum()>0 and target[1].sum()>0
        samples[suite]=(rgb,target)
    checks['six_suites_two_labels_alignment_and_replay']='PASS'
    loss_fn=TwoHeadLoss()
    z=torch.zeros((1,2,4,4),requires_grad=True);empty=torch.zeros_like(z);loss,_=loss_fn(z,empty)
    assert abs(float(loss.detach())-(np.log(2)+.5))<1e-6
    loss.backward();assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    perfect=segmentation(np.array([[1,0],[0,1]],bool),np.array([[1,0],[0,1]],bool));assert perfect['dice']==perfect['recall']==1.
    empty_bad=segmentation(np.ones((2,2),bool),np.zeros((2,2),bool));assert empty_bad['precision']==0. and empty_bad['recall']==1.
    # Independent head truth and gates: perfect synthetic predictions must pass,
    # and losing missing predictions must not pass via easy negative images.
    fixture_rows=[]
    for record in data.val:
        # Synthetic arrays, not validation model inference or GT-image reads.
        gt_s=np.zeros((4,4),bool);gt_m=np.zeros((4,4),bool)
        if record['suite'] in ('scratch','combined'):gt_s[0,0]=1
        if record['suite'] in ('missing','combined'):gt_m[2,2]=1
        r=dict(id=record['id'],suite=record['suite'],severity=record['severity'])
        for head,gt in [('scratch',gt_s),('missing',gt_m),('union',gt_s|gt_m)]:r.update({head+'_'+k:v for k,v in segmentation(gt,gt).items()})
        r['union_outside_synthetic_gt_fraction']=0.;fixture_rows.append(r)
    assert summarize(fixture_rows)['numerically_eligible']
    for r in fixture_rows:
        if r['suite'] in ('missing','combined'):
            r.update({ 'missing_'+k:v for k,v in segmentation(np.zeros((4,4),bool),np.eye(4,dtype=bool)).items()})
    assert not summarize(fixture_rows)['numerically_eligible']
    checks['loss_empty_gradient_metrics_and_no_negative_inflation']='PASS'
    # Actual architecture, 512-pixel train tensors, real backward, no optimizer.
    selected=['combined'] if device.type=='cpu' else ['combined','missing','scratch','clean']
    batch=len(selected);images=torch.stack([torch.from_numpy(normalized(samples[s][0])) for s in selected]).to(device)
    targets=torch.stack([torch.from_numpy(samples[s][1]) for s in selected]).to(device)
    model.train();model.zero_grad(set_to_none=True)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    amp=device.type=='cuda' and torch.cuda.is_bf16_supported()
    step_seconds=[]
    for _ in range(3 if device.type=='cuda' else 1):
        model.zero_grad(set_to_none=True)
        if device.type=='cuda':torch.cuda.synchronize(device)
        step_start=time.perf_counter()
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):logits=model(images);full_loss,terms=loss_fn(logits,targets)
        assert logits.shape==(batch,2,512,512) and torch.isfinite(full_loss)
        full_loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if device.type=='cuda':torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter()-step_start)
    head=model.segmentation_head[0]
    head_grad=[float(head.weight.grad[i].norm()) for i in range(2)]
    assert all(v>0 and np.isfinite(v) for v in head_grad)
    checks.update(fullsize_forward_backward='PASS',head_gradient_norms_after_clip=head_grad,
                  fullsize_loss=float(full_loss.detach()),loss_terms=terms,gradient_norm_before_clip=float(norm))
    inference_seconds=[];model.eval()
    for _ in range(3):
        if device.type=='cuda':torch.cuda.synchronize(device)
        pred_start=time.perf_counter();probability(model,samples['combined'][0],device)
        if device.type=='cuda':torch.cuda.synchronize(device)
        inference_seconds.append(time.perf_counter()-pred_start)
    estimate=(600/batch)*float(np.median(step_seconds[1:] or step_seconds))+1600*float(np.median(inference_seconds[1:]))
    report=dict(status='PASS',protocol_sha256=verify_bundle(),bundle_sha256=digest(HERE/'BUNDLE_MANIFEST.json'),environment=env,
        device=device_name,batch_size=batch,precision='bfloat16' if amp else 'float32',transfer=transfer,
        checks=checks,elapsed_seconds=time.perf_counter()-start,optimizer_steps=0,checkpoint_written=False,
        candidate_quality_measured=False,test_payloads_opened=0,
        measured_forward_backward_seconds=step_seconds,measured_inference_seconds=inference_seconds,
        epoch_compute_estimate_seconds=estimate,estimate_scope='Compute estimate excludes data decode/augmentation, optimizer, PNG encoding, CSV, checkpoint IO; first-epoch timing is authoritative',
        gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    out.mkdir(parents=True,exist_ok=True);write_json(out/('contract_smoke_'+device_name+'.json'),report);print(json.dumps(report,indent=2))
