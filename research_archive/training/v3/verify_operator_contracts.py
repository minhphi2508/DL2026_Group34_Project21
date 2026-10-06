"""CPU-only operator/package fixtures, no neural training or GPU dependency."""
import contextlib,io,json,tempfile,zipfile
from pathlib import Path
from unittest.mock import patch
import return_outputs as pack
import run_rtx
from data import HERE,digest,verify_bundle

def check():
    checks={}
    # Fixtures live in a temporary directory; real experiment checkpoints untouched.
    with tempfile.TemporaryDirectory(prefix='twohead_operator_fixtures_') as temp:
        root=Path(temp);(root/'assets').mkdir()
        for name in ('protocol.json','BUNDLE_MANIFEST.json','README_VI.md'):(root/name).write_text('{}',encoding='utf-8')
        run=root/'runs/fixture';run.mkdir(parents=True)
        (run/'failure.json').write_text('{"status":"FAILED_FIXTURE"}',encoding='utf-8')
        with patch.object(pack,'HERE',root):
            with contextlib.redirect_stdout(io.StringIO()):result=pack.pack_outputs('fixture')
            with zipfile.ZipFile(result['light']['path']) as z:
                assert 'evidence/failure.json' in z.namelist() and 'SELECTED_RESEARCH_CHECKPOINT.pt' not in z.namelist()
                hashes=json.loads(z.read('OUTPUT_SHA256.json'));assert set(hashes)==set(z.namelist())-{'OUTPUT_SHA256.json'}
            checks['failure_light_without_selection']='PASS'
            fake=run/'best_unconstrained.pt';fake.write_bytes(b'FIXTURE_ONLY_NOT_A_MODEL')
            masks=run/'epochs/epoch_001/predicted_masks.zip';masks.parent.mkdir(parents=True);masks.write_bytes(b'FIXTURE_ONLY_NOT_MASK_ZIP')
            (run/'selection.json').write_text(json.dumps(dict(selected_checkpoint=fake.name,selected_masks='epochs/epoch_001/predicted_masks.zip',numeric_gate_passed=False)),encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):result=pack.pack_outputs('fixture')
            with zipfile.ZipFile(result['light']['path']) as z:
                assert z.read('SELECTED_RESEARCH_CHECKPOINT.pt')==fake.read_bytes()
                assert z.read('SELECTED_PREDICTED_MASKS.zip')==masks.read_bytes()
            with zipfile.ZipFile(result['full_resume_backup']['path']) as z:
                assert 'candidate_v3_pair_replay/runs/fixture/best_unconstrained.pt' in z.namelist()
            checks['one_diagnostic_checkpoint_matching_epoch_masks_and_resume_layout']='PASS'
            (run/'selection.json').write_text(json.dumps(dict(selected_checkpoint='../escape.pt',selected_masks='epochs/epoch_001/predicted_masks.zip')),encoding='utf-8')
            try:pack.pack_outputs('fixture')
            except ValueError:pass
            else:raise AssertionError('Unsafe selection accepted')
            checks['selection_path_escape_rejected']='PASS'
        dry=root/'runs/dry'
        with patch.object(run_rtx,'HERE',root),patch('subprocess.Popen',side_effect=AssertionError('Dryrun spawned a child')):
            with contextlib.redirect_stdout(io.StringIO()):assert run_rtx.run_actions(root/'data','dry',True)==0
        assert (dry/'operator_plan.json').exists();checks['dryrun_no_children_no_training']='PASS'
        calls=[]
        class FailedProcess:
            stdout=iter(['fixture failure before GPU\n'])
            def wait(self):return 7
        def fail(cmd,**kwargs):calls.append(cmd);return FailedProcess()
        def fixture_pack(name):
            assert name=='failed';return {}
        with patch.object(run_rtx,'HERE',root),patch('subprocess.Popen',side_effect=fail),patch.object(pack,'pack_outputs',side_effect=fixture_pack):
            with contextlib.redirect_stdout(io.StringIO()):assert run_rtx.run_actions(root/'data','failed')==1
        assert len(calls)==1 and calls[0][2]=='preflight'
        status=json.loads((root/'runs/failed/operator_status.json').read_text());assert status['status']=='FAILED_RETURN_EVIDENCE'
        checks['failure_stops_before_smoke_train_and_still_packs']='PASS'
        resumed=root/'runs/resumed';resumed.mkdir();(resumed/'last_completed.pt').write_bytes(b'FIXTURE')
        with patch.object(run_rtx,'HERE',root),patch('subprocess.Popen',side_effect=AssertionError('Dryrun spawned child')):
            with contextlib.redirect_stdout(io.StringIO()):run_rtx.run_actions(root/'data','resumed',True)
        assert json.loads((resumed/'operator_plan.json').read_text())['resume'] is True
        checks['checkpoint_auto_resume_plan']='PASS'
    report=dict(status='PASS',checks=checks,protocol_sha256=verify_bundle(),bundle_sha256=digest(HERE/'BUNDLE_MANIFEST.json'),
        actual_neural_inference=False,optimizer_steps=0,test_payloads_opened=0,fixture_artifacts='Synthetic bytes never used as model-quality evidence')
    out=HERE/'runs/local_contract';out.mkdir(parents=True,exist_ok=True);(out/'operator_fixture_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':check()
