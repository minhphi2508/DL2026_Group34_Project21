"""Full validation only, with independent scratch/missing/union diagnostics."""
from __future__ import annotations
import csv, hashlib, io, json, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from data import protocol
from model import masks

def segmentation(pred,gt):
    pred=np.asarray(pred,bool);gt=np.asarray(gt,bool)
    if pred.shape!=gt.shape:raise ValueError('Extent mismatch')
    tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum())
    # Fixed conventions for empty GT; positive groups are reported separately.
    if not gt.any():
        dice=iou=precision=1. if not pred.any() else 0.;recall=1.
    else:
        dice=2*tp/(2*tp+fp+fn);iou=tp/(tp+fp+fn);precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn)
    return dict(tp=tp,fp=fp,fn=fn,dice=dice,iou=iou,precision=precision,recall=recall,
                predicted_pixels=int(pred.sum()),gt_pixels=int(gt.sum()),coverage=float(pred.mean()),empty_mask=not pred.any())

def stats(rows,head):
    if not rows:raise ValueError('Empty aggregation')
    def arr(k):return [r[head+'_'+k] for r in rows]
    tp=sum(arr('tp'));fp=sum(arr('fp'));fn=sum(arr('fn'))
    return dict(n=len(rows),sources=len({r['id'] for r in rows}),
        **{k:float(np.mean(arr(k))) for k in ('dice','iou','precision','recall','coverage')},
        micro_precision=tp/(tp+fp) if tp+fp else None,micro_recall=tp/(tp+fn) if tp+fn else None,
        tp=tp,fp=fp,fn=fn,empty_masks=sum(arr('empty_mask')),
        coverage_p95=float(np.quantile(arr('coverage'),.95)),coverage_max=max(arr('coverage')),
        recall_p05=float(np.quantile(arr('recall'),.05)),recall_below_05=sum(v<.5 for v in arr('recall')))

def summarize(rows):
    if len(rows)!=1600 or len({(r['id'],r['suite'],r['severity']) for r in rows})!=1600:raise ValueError('Incomplete or duplicate validation rows')
    groups={}
    scopes={'scratch_positive':(('scratch','combined'),'scratch'),
            'missing_positive':(('missing','combined'),'missing'),
            'union_positive':(('scratch','missing','combined'),'union'),
            'union_negative':(('clean','noise','age_quality'),'union')}
    for label,(suites,head) in scopes.items():
        rr=[r for r in rows if r['suite'] in suites];groups[label]=stats(rr,head)
        for severity in ('low','medium','high'):
            subset=[r for r in rr if r['severity']==severity]
            if subset:groups[label+'_'+severity]=stats(subset,head)
    for suite in ('scratch','missing','combined','noise','age_quality','clean'):
        rr=[r for r in rows if r['suite']==suite]
        for head in ('scratch','missing','union'):groups[suite+'_'+head]=stats(rr,head)
        if suite in ('missing','combined'):
            for severity in ('low','medium','high'):
                groups[suite+'_missing_'+severity]=stats([r for r in rr if r['severity']==severity],'missing')
    g=protocol()['numerical_eligibility_gates'];s=groups['scratch_positive'];m=groups['union_negative'];u=groups['union_positive']
    checks={'scratch_dice':s['dice']>=g['scratch_macro_dice_min'],
        'scratch_precision':s['precision']>=g['scratch_macro_precision_min'],
        'scratch_recall':s['recall']>=g['scratch_macro_recall_min'],
        'scratch_suite_dice':groups['scratch_scratch']['dice']>=g['scratch_suite_dice_min'],
        'union_positive_dice':u['dice']>=g['union_positive_macro_dice_min'],
        'union_negative_mean':m['coverage']<=g['union_negative_mean_coverage_max'],
        'union_negative_p95':m['coverage_p95']<=g['union_negative_p95_coverage_max'],
        'union_negative_max':m['coverage_max']<=g['union_negative_max_coverage_max']}
    for suite in ('missing','combined'):
        for sev in ('low','medium','high'):
            mm=groups[suite+'_missing_'+sev]
            checks['missing_recall_'+suite+'_'+sev]=mm['recall']>=g['missing_per_suite_severity_macro_recall_min']
            checks['missing_precision_'+suite+'_'+sev]=mm['precision']>=g['missing_per_suite_severity_macro_precision_min']
    rr=[r for r in rows if r['suite'] in ('scratch','missing','combined')]
    content_fp=float(np.mean([r['union_outside_synthetic_gt_fraction'] for r in rr]))
    checks['outside_synthetic_gt_proxy']=content_fp<=g['union_positive_mean_outside_synthetic_gt_fraction_max']
    score=.5*s['dice']+.5*groups['missing_positive']['dice']
    return dict(groups=groups,score=score,eligibility_checks=checks,numerically_eligible=all(checks.values()),
        mean_union_outside_synthetic_gt_fraction=content_fp,outside_gt_scope='Proxy only; native historical damage may be unlabeled',promotion='REQUIRES_HUMAN_AND_END_TO_END_REVIEW',
        conditions=1600,source_photos=100,test_payloads_opened=0)

def run_validation(model,data,device):
    model.eval();rows=[];pngs={};start_all=time.perf_counter();timings={'data_decode_and_hash':0.,'inference_and_mask':0.,'png_encoding':0.}
    for r in data.val:
        decode_start=time.perf_counter()
        rgb,scratch,missing=data.validation_sample(r)
        timings['data_decode_and_hash']+=time.perf_counter()-decode_start
        start=time.perf_counter();pred=masks(model,rgb,device)
        if device.type=='cuda':torch.cuda.synchronize(device)
        model_time=time.perf_counter()-start;timings['inference_and_mask']+=model_time
        row=dict(id=r['id'],split='val',suite=r['suite'],severity=r['severity'],runtime_ms=model_time*1000)
        truths={'scratch':scratch,'missing':missing,'union':scratch|missing}
        for head,gt in truths.items():
            values=segmentation(pred[head]>0,gt);row.update({head+'_'+k:v for k,v in values.items()})
            encode_start=time.perf_counter();buff=io.BytesIO();Image.fromarray(pred[head]).save(buff,format='PNG')
            pngs[f'{head}/{r["suite"]}/{r["severity"]}/{r["id"]}.png']=buff.getvalue()
            timings['png_encoding']+=time.perf_counter()-encode_start
        row['union_outside_synthetic_gt_fraction']=float(((pred['union']>0)&~truths['union']).mean())
        rows.append(row)
    summary=summarize(rows);timings['wall_total']=time.perf_counter()-start_all
    timings['metrics_and_other']=timings['wall_total']-sum(timings[k] for k in ('data_decode_and_hash','inference_and_mask','png_encoding'))
    summary['validation_timing_seconds']=timings
    return rows,summary,pngs

def save_validation(directory,rows,summary,pngs=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'per_case.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if pngs is not None:
        hashes={k:hashlib.sha256(v).hexdigest() for k,v in pngs.items()}
        with zipfile.ZipFile(directory/'predicted_masks.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for k,v in pngs.items():z.writestr(k,v)
            z.writestr('MASK_SHA256.json',json.dumps(hashes,sort_keys=True))
