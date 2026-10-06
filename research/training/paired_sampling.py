"""DRAFT paired-low sampler only. No model initialization, training or operator.
Uses a pinned FrozenData-compatible TRAIN reader; root owns integration/sealing."""
import numpy as np
SEED=20261005

EXPECTED_POLICY={
    'name':'paired_low_same_context_v1','sampler_seed':20261005,
    'pairs_per_suite':45,'positive_suites':['missing','combined'],
    'paired_negative_suites':{'missing':'clean','combined':'noise'},
    'positive_slot_counts':{'medium':22,'high':23},'new_positive_severity':'low',
    'noise_negative_per_severity':15,'ordinary_slots_exactly_unchanged':420,
    'unique_sources_per_epoch':510,'low_positive_counts':{'missing':85,'combined':95},
    'paired_crop_and_augmentation':'same_pre_degradation_RGB_and_shared_dihedral_no_interpolation',
    'source_selection':'deterministic_train_schedule_only_no_validation_mining',
    'parent_optimizer_state_used':False,
}
EXPECTED_MIXTURE={'scratch':180,'missing':120,'combined':150,'clean':50,'noise':50,'age_quality':50}

def verify_policy(data):
    p=data.p;t=p['training']
    if t.get('sampling_policy')!=EXPECTED_POLICY:raise ValueError('Paired sampler differs from explicit protocol policy')
    if p['seed']!=20261004 or p['input_size']!=512 or t['generation_epoch_offset']!=20:raise ValueError('Changed frozen generation seed, crop size or warmstart offset')
    if t['photos']!=600 or t['mixture_counts']!=EXPECTED_MIXTURE or len(data.train)!=600:raise ValueError('Paired sampler requires the pinned600-source mixture')
    return t['sampling_policy']

def draft_schedule(data,epoch):
    """CPU DRAFT ONLY: keep mixture; 90 low positive + 90 crop-matched negatives."""
    verify_policy(data)
    if not isinstance(epoch,int) or epoch<20:raise ValueError('Fresh generation epoch must be at least20')
    baseline=data.ordinary_schedule(epoch) if hasattr(data,'ordinary_schedule') else data.schedule(epoch)
    items=[dict(index=i,suite=s,severity=v,pair_id=None,role='ordinary',generation_epoch=epoch) for i,s,v in baseline]
    rng=np.random.default_rng(np.random.SeedSequence([SEED,epoch,90]))
    positives={s:[j for j,x in enumerate(items) if x['suite']==s] for s in ('missing','combined')}
    clean=[j for j,x in enumerate(items) if x['suite']=='clean'];rng.shuffle(clean)
    noise=[]
    for sev in ('low','medium','high'):
        candidates=[j for j,x in enumerate(items) if x['suite']=='noise' and x['severity']==sev]
        rng.shuffle(candidates);noise.extend(candidates[:15])
    rng.shuffle(noise)
    for suite,negative_slots in [('missing',clean[:45]),('combined',noise)]:
        pair_slots=[]
        for severity,count in [('medium',22),('high',23)]:
            selected=[j for j in positives[suite] if items[j]['severity']==severity]
            rng.shuffle(selected);pair_slots.extend(selected[:count])
        rng.shuffle(pair_slots)
        for n,(pj,nj) in enumerate(zip(pair_slots,negative_slots)):
            pos=items[pj];neg=items[nj];pair_id=f'e{epoch:02d}_{suite}_{n:02d}'
            pos.update(severity='low',pair_id=pair_id,role='positive')
            neg.update(index=pos['index'],pair_id=pair_id,role='negative',crop_suite=suite)
    # Keep the original slot order. Every ordinary slot stays identical to v2.
    return items

def render_sample(data,item,epoch):
    verify_policy(data)
    epoch=item.get('generation_epoch',epoch)
    row=data.train[item['index']];suite=item['suite'];sev=item['severity'];d=data.degradation
    seed=d.stable_seed('missing_mask_twohead_v1',row['dataset_path'],epoch,suite,base=data.p['seed'])
    crop_seed=d.stable_seed('missing_mask_twohead_v1',row['dataset_path'],epoch,item.get('crop_suite',suite),base=data.p['seed'])
    original=data.open_image(row['dataset_path'],training=True,expected=row['dataset_sha256'])
    crop=d.deterministic_crop(original,512,crop_seed)
    rgb0=np.array(crop)
    if suite=='clean':
        rgb=rgb0.copy();scratch=missing=np.zeros((512,512),bool)
    else:
        damaged=d.degrade(crop,suite,sev,seed,data.cfg)
        rgb=np.array(damaged['image']);scratch=np.array(damaged['scratch_mask'])>0;missing=np.array(damaged['missing_mask'])>0
    augment_seed=crop_seed if item['pair_id'] else seed
    rng=np.random.default_rng(d.stable_seed(augment_seed,'augmentation',base=data.p['seed']))
    rotation=int(rng.integers(0,4));flip_x=bool(rng.random()<.5);flip_y=bool(rng.random()<.5)
    def transform(x):
        a=np.rot90(x,rotation)
        if flip_x:a=a[:,::-1]
        if flip_y:a=a[::-1]
        return np.ascontiguousarray(a)
    target=np.ascontiguousarray(transform(np.stack((scratch & ~missing,missing),axis=0).transpose(1,2,0)).transpose(2,0,1),dtype=np.float32)
    meta=dict(source=row['dataset_path'],suite=suite,severity=sev,seed=seed,generation_epoch=epoch,pair_id=item['pair_id'],role=item['role'],crop_seed=crop_seed,degradation_seed=seed,rotation=rotation,flip_x=flip_x,flip_y=flip_y)
    return transform(rgb),target,transform(rgb0),meta
