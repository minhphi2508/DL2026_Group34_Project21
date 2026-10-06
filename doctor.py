"""Check runtime imports, execute a device kernel, and verify pinned assets."""
import argparse,json,platform,sys
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto');p.add_argument('--skip-models',action='store_true');a=p.parse_args()
    import torch,torchvision,cv2,dlib,segmentation_models_pytorch,skimage
    from restoration.pipeline import choose_device
    from restoration.assets import verify_models,verify_source
    device=choose_device(a.device)
    result=dict(status='PASS',python=sys.version.split()[0],platform=platform.platform(),device=device,torch=torch.__version__,torchvision=torchvision.__version__,
        gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,source_files=verify_source(),models=None if a.skip_models else len(verify_models()))
    print(json.dumps(result,indent=2));return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:print('CHECK FAILED: '+str(error),file=sys.stderr);raise SystemExit(1)
