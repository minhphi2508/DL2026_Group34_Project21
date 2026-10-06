"""One explicit RTX launch: preflight, kernel/backward smoke, train, evaluate, pack."""
import argparse,json,re,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def run_actions(data_root,run_name,dry_run=False,resume=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}',run_name):raise ValueError('Unsafe run name')
    out=HERE/'runs'/run_name;out.mkdir(parents=True,exist_ok=True)
    actions=['preflight','smoke','train']
    # Full predictions already cached at the selected best epoch. Reuse them;
    # no extra neural1600 pass just to export masks after training.
    if (out/'last_completed.pt').exists() and not (out/'completion.json').exists():resume=True
    plan=dict(actions=actions,data_root=str(Path(data_root).resolve()),run_name=run_name,device='cuda',resume=resume,
        training_wall_budget_seconds=3600,epochs_max=15,final_test='FORBIDDEN',promotion='NEVER_AUTOMATIC',
        returns=['RETURN_LIGHT_'+run_name+'.zip','BACKUP_FULL_RESUME_'+run_name+'.zip'])
    (out/'operator_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    if dry_run:
        print(json.dumps(dict(status='DRY_RUN_NO_NEURAL_NO_TRAINING',**plan),indent=2));return 0
    state=dict(status='RUNNING',completed_actions=[],started_unix=time.time(),test_payloads_opened=0)
    code=0
    try:
        for action in actions:
            if action=='train' and (out/'completion.json').is_file():
                print('Training completion already present; preserving checkpoint and packaging cached selected masks',flush=True);continue
            cmd=[sys.executable,str(HERE/'runner.py'),action,'--data-root',str(Path(data_root).resolve()),'--run-name',run_name,'--device','cuda']
            if action=='train' and resume:cmd.append('--resume')
            print('STEP '+action,flush=True)
            with (out/'operator_console.log').open('a',encoding='utf-8') as log:
                process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',cwd=HERE)
                for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                rc=process.wait()
            if rc:raise RuntimeError(action+' failed with exit '+str(rc))
            state['completed_actions'].append(action)
            if action=='smoke':
                report=json.loads((out/'contract_smoke_cuda.json').read_text(encoding='utf-8'))
                print('Compute estimate/epoch: '+str(round(report['epoch_compute_estimate_seconds'],1))+'s, excludes data/PNG/checkpoint IO; research budget1h at epoch boundaries, max15epochs.',flush=True)
        state['status']='COMPLETED_RESEARCH_REQUIRES_REVIEW'
        state['numeric_eligible']=(out/'best_eligible.pt').is_file()
    except Exception as exc:
        state.update(status='FAILED_RETURN_EVIDENCE',error=repr(exc));code=1
    finally:
        state['elapsed_seconds']=time.time()-state['started_unix'];(out/'operator_status.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        from return_outputs import pack_outputs
        pack_outputs(run_name)
    return code

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',default=str(HERE.parent/'data'));p.add_argument('--run-name',default='candidate_v3_pair_replay');p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    raise SystemExit(run_actions(a.data_root,a.run_name,a.dry_run,a.resume))
