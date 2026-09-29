from __future__ import annotations

import gc, hashlib, importlib.util, json, math, os, shutil, subprocess, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from tqdm.auto import tqdm

ROOT=Path('/kaggle/working/EviCT'); WORK=Path('/kaggle/working')
CFG=ROOT/'config/notebook06_semantic_branch.json'; BASECFG=ROOT/'config/segformer_b1_baseline.json'
MODEL=ROOT/'src/evict/models/semantic_segformer.py'; METRICS=ROOT/'src/evict/metrics.py'
STATE=ROOT/'STATE.md'; SEL=ROOT/'manifests/source_selection_slices.csv'; SPLITS=ROOT/'manifests/splits.csv'
AUD=ROOT/'artifacts/audit'; LARGE=ROOT/'artifacts/large/notebook06'; TABLES=ROOT/'tables'; REPORTS=ROOT/'reports'
HF=WORK/'hf_nb06'; PROTO=AUD/'notebook06_text_prototypes.pt'; PROTO_META=AUD/'notebook06_text_prototype_manifest.json'
TEXT_PIN=ROOT/'config/notebook06_text_model_pin.json'; PROMPTS=ROOT/'config/notebook06_prompts.json'; SMOKE=AUD/'notebook06_semantic_gpu_smoke.json'
FINAL=AUD/'notebook06_semantic_ablation.json'; FINAL_CSV=TABLES/'notebook06_semantic_ablation.csv'; FINAL_REPORT=REPORTS/'notebook06_semantic_branch_report.md'
CACHE_TAR=WORK/'EViCT_Notebook03_Cache.tar'; CACHE_PROBE=ROOT/'cache/segdb2/images/coronacases_003.npy'
CACHE_URL='https://github.com/itsCodeBakery/EviCT/releases/download/evict-project-durable-backup-20260927/EViCT_Notebook03_Cache.tar'
CACHE_SHA='d88e786cd467816d6cb446333385918016946a83cdd3c258b3011f458fbd3fc4'
SEEDS=[17,42,2026]; VARIANTS=['real_text','swapped_text','random_prototypes']
MAX=5000; WARM=200; VAL=250; REC=50; PAT=8; MICRO=4; ACC=4; ENC_LR=1e-4; HEAD_LR=3e-4; WD=.01; AUX=.1
ROLL={1000,2000,3000,4000}
for d in [AUD,LARGE,TABLES,REPORTS,HF]: d.mkdir(parents=True,exist_ok=True)

def loadmod(path,name):
    s=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
base=loadmod(ROOT/'scripts/nb05b_unet_lr_pilot_segment1.py','nb06base')
nb05e=loadmod(ROOT/'scripts/nb05e_segct_clip_visual_adaptation.py','nb06nb05e')

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return base.sha256_file(Path(p))
def jhash(o): return base.stable_json_hash(o)
def train_manifest(seed): return ROOT/f'manifests/slice_loaders/seed_{seed}/b100/labeled_slices.csv'
def rp(v,s):
    d=LARGE/v/f'seed_{s}'; d.mkdir(parents=True,exist_ok=True); pre=f'notebook06_{v}_seed{s}'
    return {'d':d,'last':d/'last.pt','rec':d/'recovery.pt','best':d/'best.pt','train':AUD/f'{pre}_train_log.csv','sel':AUD/f'{pre}_selection_metrics.csv','final':AUD/f'{pre}_final_durable.json'}

def state_ok():
    t=STATE.read_text(); assert ('NOTEBOOK_05E_SEGCT_CLIP_VISUAL_ADAPTATION_THREE_SEED_FROZEN' in t or 'NOTEBOOK_06_' in t)
    assert 'Calibration accessed:\n\nNO' in t and 'Target / MedSeg accessed:\n\nNO' in t and 'Target lock:\n\nACTIVE' in t

def ensure_cache():
    if CACHE_PROBE.exists(): print('✓ Notebook-03 cache        : PRESENT'); nb05e.verify_source_cache(); return
    misplaced=WORK/'cache/segdb2'
    if misplaced.exists():
        target=ROOT/'cache/segdb2'; target.parent.mkdir(parents=True,exist_ok=True); shutil.rmtree(target,ignore_errors=True); shutil.move(str(misplaced),str(target)); nb05e.verify_source_cache(); return
    if CACHE_TAR.exists() and sha(CACHE_TAR)!=CACHE_SHA: CACHE_TAR.unlink()
    if not CACHE_TAR.exists(): base.download_verified(CACHE_URL,CACHE_TAR,CACHE_SHA)
    base.safe_extract_tar(CACHE_TAR,ROOT); assert CACHE_PROBE.exists(); nb05e.verify_source_cache(); CACHE_TAR.unlink(missing_ok=True)

def manifests():
    a=pd.read_csv(train_manifest(17)); sel=pd.read_csv(SEL); cols=['case_id','image_array_index','image_path','infection_cache_path','valid_mask_path']
    ref=a[cols].sort_values(['case_id','image_array_index']).reset_index(drop=True)
    for s in [42,2026]: pd.testing.assert_frame_equal(ref,pd.read_csv(train_manifest(s))[cols].sort_values(['case_id','image_array_index']).reset_index(drop=True),check_dtype=False)
    assert len(set(a.case_id.astype(str)))==12 and len(set(sel.case_id.astype(str)))==4
    print('✓ Frozen fitting/selection : 12 / 4 cases'); return sel

def mit_snapshot():
    from huggingface_hub import snapshot_download
    c=json.loads(BASECFG.read_text()); r=c['checkpoint']; p=Path(snapshot_download(repo_id=r['repository'],revision=r['revision'],cache_dir=str(HF/'mit'),allow_patterns=['config.json','pytorch_model.bin']))
    assert sha(p/'pytorch_model.bin')==r['weight_sha256']; print('✓ MiT-B1 pin               :',r['revision'][:12]); return p

def text_prototypes():
    c=json.loads(CFG.read_text()); ph=jhash(c['prompt_bank'])
    if PROTO.exists() and PROTO_META.exists() and TEXT_PIN.exists() and PROMPTS.exists():
        m=json.loads(PROTO_META.read_text())
        if m.get('status')=='FROZEN' and m.get('prompt_bank_hash')==ph and m.get('prototype_file_sha256')==sha(PROTO):
            print('✓ BiomedCLIP prototypes    : REUSED'); return torch.load(PROTO,map_location='cpu',weights_only=False)
    from huggingface_hub import HfApi,snapshot_download
    import open_clip
    repo=c['text_model']['repository']; rev=HfApi().model_info(repo).sha
    base.atomic_text(TEXT_PIN,json.dumps({'repository':repo,'revision':rev,'resolved_utc':now()},indent=2)+'\n')
    base.atomic_text(PROMPTS,json.dumps(c['prompt_bank'],indent=2)+'\n')
    snap=Path(snapshot_download(repo_id=repo,revision=rev,cache_dir=str(HF/'biohub'),local_dir=str(HF/'biosnap')))
    name=f'local-dir:{snap}'; model,_=open_clip.create_model_from_pretrained(name,device='cpu',precision='fp32'); tok=open_clip.get_tokenizer(name); model.eval()
    ps=c['prompt_bank']['foreground']+c['prompt_bank']['background']
    with torch.no_grad(): e=model.encode_text(tok(ps),normalize=True).float().cpu()
    fg=torch.nn.functional.normalize(torch.nn.functional.normalize(e[:3],dim=1).mean(0),dim=0); bg=torch.nn.functional.normalize(torch.nn.functional.normalize(e[3:],dim=1).mean(0),dim=0)
    g=torch.Generator().manual_seed(606); rf=torch.nn.functional.normalize(torch.randn(fg.numel(),generator=g),dim=0); rb=torch.nn.functional.normalize(torch.randn(fg.numel(),generator=g),dim=0)
    out={'revision':rev,'embedding_dim':int(fg.numel()),'prompt_bank_hash':ph,'real_foreground':fg,'real_background':bg,'random_foreground':rf,'random_background':rb,'random_seed':606}
    base.atomic_torch_save(out,PROTO); meta={'timestamp_utc':now(),'status':'FROZEN','text_model_revision':rev,'embedding_dim':int(fg.numel()),'prompt_bank_hash':ph,'prototype_file_sha256':sha(PROTO),'calibration_accessed':False,'target_accessed':False}
    base.atomic_text(PROTO_META,json.dumps(meta,indent=2)+'\n'); base.git_sync('Freeze Notebook 06 BiomedCLIP text prototypes and prompt bank')
    del model,tok,e; gc.collect(); shutil.rmtree(HF/'biosnap',ignore_errors=True); shutil.rmtree(HF/'biohub',ignore_errors=True)
    print('✓ BiomedCLIP prototypes    : FROZEN'); return out

def pair(proto,v):
    if v=='real_text': return proto['real_foreground'],proto['real_background']
    if v=='swapped_text': return proto['real_background'],proto['real_foreground']
    return proto['random_foreground'],proto['random_background']

def sched(step):
    if step<WARM: return (step+1)/WARM
    q=min(max((step-WARM)/(MAX-WARM),0),1); return .5*(1+math.cos(math.pi*q))

def split_hash(seed): return jhash({'splits':sha(SPLITS),'train':sha(train_manifest(seed)),'selection':sha(SEL)})

def build(snap,proto,v,device):
    from evict.models.semantic_segformer import EviCTSemanticSegFormerB1
    fg,bg=pair(proto,v); return EviCTSemanticSegFormerB1(str(snap),fg,bg,256,.1,.1).to(device)

def smoke(snap,proto,device):
    if SMOKE.exists() and json.loads(SMOKE.read_text()).get('status')=='PASS': print('✓ GPU smoke                 : REUSED'); return
    from evict.models.semantic_segformer import masked_supervised_loss,semantic_patch_auxiliary_loss
    m=build(snap,proto,'real_text',device); x=torch.rand(1,1,336,336,device=device); y=torch.zeros_like(x); y[:,:,120:180,140:210]=1; v=torch.ones_like(x)
    o=m(base.normalize_batch(x,device)); ls=masked_supervised_loss(o['logits'],y,v); la=semantic_patch_auxiliary_loss(o['semantic_quarter_logits'],y,v); loss=ls['loss']+AUX*la['loss']; loss.backward()
    assert all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None)
    a={'timestamp_utc':now(),'status':'PASS','gpu':torch.cuda.get_device_name(0),'text_dim':m.text_dim,'alpha':float(o['alpha'].item()),'semantic_grid':list(o['semantic_quarter_logits'].shape),'calibration_accessed':False,'target_accessed':False}
    base.atomic_text(SMOKE,json.dumps(a,indent=2)+'\n'); base.git_sync('Complete Notebook 06 semantic branch GPU smoke test'); del m,x,y,v,o,ls,la,loss; gc.collect(); torch.cuda.empty_cache(); print('✓ GPU smoke                 : PASS')

def ckp(model,opt,sch,sampler,v,s,step,best,best_step,pat,last_val,images,ch,mh,xh,ph,sh):
    return {'stage':'NOTEBOOK_06','variant':v,'seed':s,'global_step':step,'best_score':best,'best_step':best_step,'patience_count':pat,'last_validation_step':last_val,'images_seen':images,'student_state_dict':{k:z.detach().cpu() for k,z in model.state_dict().items()},'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':sch.state_dict(),'rng_state':base.capture_rng_state(),'sampler_state':sampler.state_dict(),'config_hash':ch,'model_code_hash':mh,'metrics_code_hash':xh,'prototype_file_sha256':ph,'split_hash':sh,'calibration_accessed':False,'target_accessed':False,'timestamp_utc':now()}
def save(path,**kw): base.atomic_torch_save(ckp(**kw),path)
def check(q,v,s,ch,mh,xh,ph,sh):
    assert q['stage']=='NOTEBOOK_06' and q['variant']==v and int(q['seed'])==s and q['config_hash']==ch and q['model_code_hash']==mh and q['metrics_code_hash']==xh and q['prototype_file_sha256']==ph and q['split_hash']==sh and not q['calibration_accessed'] and not q['target_accessed']

def archive(v,s,p,step,final=False):
    # Keep one stable rolling asset name so each milestone replaces the previous
    # release asset instead of accumulating multi-GB historical copies.
    stem=(f'EViCT_Notebook06_{v}_seed{s}_final_step{step}_Recovery' if final else f'EViCT_Notebook06_{v}_seed{s}_Rolling_Recovery')
    tar=WORK/f'{stem}.tar'; sp=Path(str(tar)+'.sha256'); tar.unlink(missing_ok=True); sp.unlink(missing_ok=True)
    slim_best=WORK/f'nb06_{v}_seed{s}_best_model_tmp.pt'; slim_best.unlink(missing_ok=True)
    if p['best'].exists():
        q=torch.load(p['best'],map_location='cpu',weights_only=False)
        base.atomic_torch_save({
            'stage':'NOTEBOOK_06','variant':v,'seed':s,
            'global_step':int(q['global_step']),'best_score':float(q['best_score']),
            'best_step':int(q['best_step']),'student_state_dict':q['student_state_dict'],
            'calibration_accessed':False,'target_accessed':False,
        },slim_best)
        del q
    with tarfile.open(tar,'w') as t:
        if slim_best.exists(): t.add(slim_best,arcname=str(Path('EviCT')/p['best'].relative_to(ROOT)))
        if p['rec'].exists(): t.add(p['rec'],arcname=str(Path('EviCT')/p['rec'].relative_to(ROOT)))
        for src in [p['train'],p['sel']]:
            if src.exists(): t.add(src,arcname=str(Path('EviCT')/src.relative_to(ROOT)))
    slim_best.unlink(missing_ok=True); d=sha(tar); base.atomic_text(sp,d); return tar,sp,d

def release(token,v,s,p,step,final,best,best_step):
    tar,sp,d=archive(v,s,p,step,final); tag=f'evict-nb06-{v}-seed{s}-'+(f'final-step{step}' if final else 'rolling')
    rel,headers,api=base.create_or_get_release(token,tag,f'EviCT Notebook 06 — {v} seed {s} — {"final" if final else "rolling"}',f'Notebook 06 {v}, seed={s}, step={step}, best Dice={best:.8f}@{best_step}. Calibration and target/MedSeg were not accessed.')
    base.upload_asset(release=rel,headers=headers,api=api,file_path=tar,content_type='application/x-tar'); base.upload_asset(release=rel,headers=headers,api=api,file_path=sp,content_type='text/plain'); tar.unlink(); sp.unlink(); return rel['html_url'],d

def restore_roll(token,v,s,p):
    if p['rec'].exists() or p['final'].exists(): return
    tag=f'evict-nb06-{v}-seed{s}-rolling'; h={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json'}; r=requests.get(f'https://api.github.com/repos/itsCodeBakery/EviCT/releases/tags/{tag}',headers=h,timeout=60)
    if r.status_code==404: return
    r.raise_for_status(); assets={a['name']:a for a in r.json().get('assets',[])}; tn=next((n for n in assets if n.endswith('_Recovery.tar')),None); sn=next((n for n in assets if n.endswith('.tar.sha256')),None)
    if not tn or not sn: return
    expected=requests.get(assets[sn]['browser_download_url'],headers=h,timeout=60).text.strip(); dest=WORK/tn; base.download_verified(assets[tn]['browser_download_url'],dest,expected); base.safe_extract_tar(dest,WORK); dest.unlink(missing_ok=True); assert p['rec'].exists(); print(f'✓ Restored rolling         : {v} seed{s}')

def run_one(v,s,snap,proto,sel,token,ch,mh,xh,ph,device):
    p=rp(v,s)
    if p['final'].exists():
        a=json.loads(p['final'].read_text());
        if a.get('status')=='DURABLE_COMPLETE': print(f'✓ {v} seed{s}              : SKIP COMPLETE'); return a
    restore_roll(token,v,s,p)
    from evict.models.semantic_segformer import masked_supervised_loss,semantic_patch_auxiliary_loss
    from evict.metrics import CaseMetricAccumulator,binary_segmentation_metrics,macro_case_summary,metrics_from_confusion
    train=pd.read_csv(train_manifest(s)); base.reset_rng(s); store=base.FittingCaseStore(train); sampler=base.PatientUniformSampler(store,s); model=build(snap,proto,v,device)
    ep=list(model.encoder.parameters()); hp=[p0 for n,p0 in model.named_parameters() if not n.startswith('encoder.')]; opt=torch.optim.AdamW([{'params':ep,'lr':ENC_LR},{'params':hp,'lr':HEAD_LR}],weight_decay=WD); sch=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=sched)
    sh=split_hash(s); step=0; best=-1.; best_step=0; pat=0; last_val=0; images=0; tr=[] if not p['train'].exists() else pd.read_csv(p['train']).to_dict('records'); sl=[] if not p['sel'].exists() else pd.read_csv(p['sel']).to_dict('records')
    cand=[]
    for path in [p['last'],p['rec']]:
        if path.exists(): q=torch.load(path,map_location='cpu',weights_only=False); check(q,v,s,ch,mh,xh,ph,sh); cand.append((int(q['global_step']),path)); del q
    if cand:
        _,path=max(cand); q=torch.load(path,map_location='cpu',weights_only=False); check(q,v,s,ch,mh,xh,ph,sh); model.load_state_dict(q['student_state_dict']); opt.load_state_dict(q['optimizer_state_dict']);
        for st in opt.state.values():
            for k,z in list(st.items()):
                if torch.is_tensor(z): st[k]=z.to(device)
        sch.load_state_dict(q['scheduler_state_dict']); sampler.load_state_dict(q['sampler_state']); base.restore_rng_state(q['rng_state']); step=int(q['global_step']); best=float(q['best_score']); best_step=int(q['best_step']); pat=int(q['patience_count']); last_val=int(q['last_validation_step']); images=int(q['images_seen']); tr=[r for r in tr if int(r['step'])<=step]; sl=[r for r in sl if int(r['step'])<=last_val]; print(f'✓ resumed {v} seed{s}      : step {step}'); del q
    else: print(f'✓ new {v} seed{s}          : step 0')
    def sv(path): save(path,model=model,opt=opt,sch=sch,sampler=sampler,v=v,s=s,step=step,best=best,best_step=best_step,pat=pat,last_val=last_val,images=images,ch=ch,mh=mh,xh=xh,ph=ph,sh=sh)
    bar=tqdm(total=MAX-step,desc=f'nb06 {v} s{s}',unit='update'); stop='MAX_UPDATES'
    while step<MAX:
        model.train(); opt.zero_grad(set_to_none=True); tl=ts=tt=0.; selected=0
        for _ in range(ACC):
            xn,yn,vn=sampler.sample(MICRO); x=torch.from_numpy(xn).to(device); y=torch.from_numpy(yn).to(device); vm=torch.from_numpy(vn).to(device); x,y,vm=base.augment(x,y,vm); o=model(base.normalize_batch(x,device)); sup=masked_supervised_loss(o['logits'],y,vm); aux=semantic_patch_auxiliary_loss(o['semantic_quarter_logits'],y,vm); loss=sup['loss']+AUX*aux['loss']; assert torch.isfinite(loss).item(); (loss/ACC).backward(); tl+=float(loss.detach().cpu())/ACC; ts+=float(sup['loss'].detach().cpu())/ACC; tt+=float(aux['loss'].detach().cpu())/ACC; selected+=aux['selected_patches']
        assert all(torch.isfinite(z.grad).all().item() for z in model.parameters() if z.grad is not None); e_lr=float(opt.param_groups[0]['lr']); h_lr=float(opt.param_groups[1]['lr']); opt.step(); sch.step(); step+=1; images+=16; alpha=float(torch.sigmoid(model.alpha_logit.detach()).cpu())
        tr.append({'step':step,'loss':tl,'supervised_loss':ts,'text_aux_loss':tt,'alpha':alpha,'selected_aux_patches':selected,'encoder_lr':e_lr,'head_lr':h_lr,'images_seen':images})
        if step%REC==0: sv(p['last']); sv(p['rec'])
        bar.update(1); bar.set_postfix(step=step,loss=f'{tl:.4f}',best=f'{best:.4f}',alpha=f'{alpha:.3f}',pat=f'{pat}/{PAT}')
        if step%VAL==0:
            met,cases,_=base.validate(model=model,device=device,selection_df=sel,masked_supervised_loss=masked_supervised_loss,CaseMetricAccumulator=CaseMetricAccumulator,binary_segmentation_metrics=binary_segmentation_metrics,metrics_from_confusion=metrics_from_confusion,macro_case_summary=macro_case_summary); last_val=step; score=float(met['macro_case_dice']); improved=score>best
            if improved: best=score; best_step=step; pat=0; sv(p['best'])
            else: pat+=1
            sl.append({'step':step,'macro_case_dice':score,'macro_case_iou':met['macro_case_iou'],'macro_case_sensitivity':met['macro_case_sensitivity'],'macro_case_specificity':met['macro_case_specificity'],'macro_slice_dice':met['macro_slice_dice'],'pooled_dice':met['pooled_dice'],'pooled_iou':met['pooled_iou'],'selection_loss':met['selection_loss'],'best_score':best,'best_step':best_step,'patience':pat,'alpha':alpha})
            pd.DataFrame(tr).to_csv(p['train'],index=False); pd.DataFrame(sl).to_csv(p['sel'],index=False); sv(p['last']); sv(p['rec']); base.atomic_text(STATE,f'# EviCT Execution State\n\n## Current stage\n\nNOTEBOOK_06_{v.upper()}_SEED_{s}_RUNNING_STEP_{step}\n\n## Best source-selection macro case Dice\n\n{best:.8f} @ {best_step}\n\n## Isolation\n\nCalibration accessed:\n\nNO\n\nTarget / MedSeg accessed:\n\nNO\n\nTarget lock:\n\nACTIVE\n'); print(f'\n[{v} seed{s}] VAL {step}: Dice={score:.8f}, best={best:.8f}@{best_step}, alpha={alpha:.4f}, patience={pat}/{PAT}'); base.git_sync(f'Notebook 06 {v} seed{s} validation step {step}')
            if step in ROLL:
                url,d=release(token,v,s,p,step,False,best,best_step); ra=AUD/f'notebook06_{v}_seed{s}_rolling_durable.json'; base.atomic_text(ra,json.dumps({'timestamp_utc':now(),'status':'DURABLE_ROLLING','variant':v,'seed':s,'step':step,'best_macro_case_dice':best,'best_step':best_step,'release_url':url,'archive_sha256':d,'calibration_accessed':False,'target_accessed':False},indent=2)+'\n'); base.git_sync(f'Record Notebook 06 {v} seed{s} rolling step {step}')
            if pat>=PAT: stop=f'EARLY_STOPPING_PATIENCE_{PAT}'; break
    bar.close(); sv(p['last']); sv(p['rec']); assert p['best'].exists(); q=torch.load(p['best'],map_location='cpu',weights_only=False); model.load_state_dict(q['student_state_dict']); del q
    met,cases,_=base.validate(model=model,device=device,selection_df=sel,masked_supervised_loss=masked_supervised_loss,CaseMetricAccumulator=CaseMetricAccumulator,binary_segmentation_metrics=binary_segmentation_metrics,metrics_from_confusion=metrics_from_confusion,macro_case_summary=macro_case_summary); alpha=float(torch.sigmoid(model.alpha_logit.detach()).cpu()); url,d=release(token,v,s,p,step,True,best,best_step)
    out={'timestamp_utc':now(),'stage':'NOTEBOOK_06','status':'DURABLE_COMPLETE','variant':v,'seed':s,'best_step':best_step,'final_step':step,'stop_reason':stop,'alpha_at_best':alpha,'macro_case_dice':float(met['macro_case_dice']),'macro_case_iou':float(met['macro_case_iou']),'macro_case_sensitivity':float(met['macro_case_sensitivity']),'macro_case_specificity':float(met['macro_case_specificity']),'macro_slice_dice':float(met['macro_slice_dice']),'pooled_dice':float(met['pooled_dice']),'pooled_iou':float(met['pooled_iou']),'release_url':url,'archive_sha256':d,'calibration_accessed':False,'target_accessed':False}
    base.atomic_text(p['final'],json.dumps(out,indent=2)+'\n'); pd.DataFrame([{'seed':s,**r} for r in cases]).to_csv(AUD/f'notebook06_{v}_seed{s}_final_case_metrics.csv',index=False); base.git_sync(f'Complete durable Notebook 06 {v} seed{s}'); print(f'✓ {v} seed{s} COMPLETE     : Dice={out["macro_case_dice"]:.8f} @ {best_step}, final={step}'); del model,opt,sch,store,sampler; gc.collect(); torch.cuda.empty_cache(); return out

def aggregate():
    b=json.loads((AUD/'notebook04_three_seed_source_baseline.json').read_text()); res=[json.loads(rp(v,s)['final'].read_text()) for v in VARIANTS for s in SEEDS]; rows=[]
    bm=float(b['aggregate']['macro_case_dice']['mean']); bsd=float(b['aggregate']['macro_case_dice']['sample_std']); bmap={int(x['seed']):x for x in b['seed_results']}; rows.append({'variant':'vision_only_segformer_b1_reference','macro_case_dice_mean':bm,'macro_case_dice_sample_sd':bsd,'seed17_dice':bmap[17]['macro_case_dice'],'seed42_dice':bmap[42]['macro_case_dice'],'seed2026_dice':bmap[2026]['macro_case_dice'],'alpha_mean':np.nan})
    for v in VARIANTS:
        rr=[x for x in res if x['variant']==v]; ds=np.array([x['macro_case_dice'] for x in rr]); rows.append({'variant':v,'macro_case_dice_mean':float(ds.mean()),'macro_case_dice_sample_sd':float(ds.std(ddof=1)),'seed17_dice':next(x['macro_case_dice'] for x in rr if x['seed']==17),'seed42_dice':next(x['macro_case_dice'] for x in rr if x['seed']==42),'seed2026_dice':next(x['macro_case_dice'] for x in rr if x['seed']==2026),'alpha_mean':float(np.mean([x['alpha_at_best'] for x in rr]))})
    pd.DataFrame(rows).to_csv(FINAL_CSV,index=False); L={r['variant']:r for r in rows}; vision=L['vision_only_segformer_b1_reference']; real=L['real_text']; sw=L['swapped_text']; rd=L['random_prototypes']; wins_vis=sum(real[f'seed{s}_dice']>vision[f'seed{s}_dice'] for s in SEEDS); wins_sw=sum(real[f'seed{s}_dice']>sw[f'seed{s}_dice'] for s in SEEDS); wins_rd=sum(real[f'seed{s}_dice']>rd[f'seed{s}_dice'] for s in SEEDS); supported=bool(real['macro_case_dice_mean']>vision['macro_case_dice_mean'] and real['macro_case_dice_mean']>sw['macro_case_dice_mean'] and real['macro_case_dice_mean']>rd['macro_case_dice_mean'] and wins_vis>=2 and wins_sw>=2 and wins_rd>=2)
    a={'timestamp_utc':now(),'stage':'NOTEBOOK_06','status':'SEMANTIC_BRANCH_FROZEN','reference_vision_only':rows[0],'variant_summaries':rows[1:],'paired_real_text_wins_vs_vision_only':wins_vis,'paired_real_text_wins_vs_swapped':wins_sw,'paired_real_text_wins_vs_random':wins_rd,'semantic_claim_rule':json.loads(CFG.read_text())['semantic_claim_rule'],'semantic_benefit_supported_by_predeclared_rule':supported,'formal_significance_test_performed':False,'calibration_accessed':False,'target_accessed':False}; base.atomic_text(FINAL,json.dumps(a,indent=2)+'\n')
    lines=['# Notebook 06 — Fixed Biomedical Text Prototypes and Semantic Branch','','| Variant | Dice mean | SD | Mean alpha |','|---|---:|---:|---:|']
    for r in rows:
        alpha_text='' if np.isnan(r['alpha_mean']) else format(r['alpha_mean'], '.4f')
        lines.append('| {} | {:.6f} | {:.6f} | {} |'.format(r['variant'],r['macro_case_dice_mean'],r['macro_case_dice_sample_sd'],alpha_text))
    lines += ['', 'Predeclared semantic-benefit rule satisfied: **{}**.'.format('YES' if supported else 'NO'), 'This is a directional repeatability rule, not a formal significance test.', 'Calibration accessed: NO.', 'Target / MedSeg accessed: NO.', '', 'Next: Notebook 07 — teacher–student low-label learning.']
    base.atomic_text(FINAL_REPORT,'\n'.join(lines)+'\n')
    base.atomic_text(STATE,f'# EviCT Execution State\n\n## Current stage\n\nNOTEBOOK_06_SEMANTIC_BRANCH_FROZEN\n\n## Semantic branch\n\nReal-text three-seed macro-case Dice: {real["macro_case_dice_mean"]:.8f}\n\nSemantic benefit supported by predeclared control rule: {"YES" if supported else "NO"}\n\n## Isolation\n\nCalibration accessed:\n\nNO\n\nTarget / MedSeg accessed:\n\nNO\n\nTarget lock:\n\nACTIVE\n\n## Next\n\nProceed to Notebook 07: teacher–student low-label learning.\n'); base.git_sync('Freeze Notebook 06 semantic branch ablation'); return a

def main():
    print('='*112); print('EVICT NOTEBOOK 06 — FIXED BIOMEDICAL TEXT PROTOTYPES + SEMANTIC BRANCH'); print('='*112)
    from kaggle_secrets import UserSecretsClient
    token=UserSecretsClient().get_secret('pushEviCT'); assert token and torch.cuda.is_available(); device=torch.device('cuda:0'); print('✓ GPU                      :',torch.cuda.get_device_name(0)); print('✓ Calibration accessed     : NO'); print('✓ Target / MedSeg accessed : NO')
    state_ok(); ensure_cache(); sel=manifests(); cfg=json.loads(CFG.read_text()); ch=jhash(cfg); mh=sha(MODEL); xh=sha(METRICS); proto=text_prototypes(); ph=sha(PROTO); snap=mit_snapshot(); smoke(snap,proto,device)
    for v in VARIANTS:
        for s in SEEDS: run_one(v,s,snap,proto,sel,token,ch,mh,xh,ph,device)
    a=aggregate(); print('\n'+'='*112); print('NOTEBOOK 06 — DURABLE COMPLETE'); print('='*112)
    for r in a['variant_summaries']: print(f"{r['variant']:<20} Dice={r['macro_case_dice_mean']:.8f} ± {r['macro_case_dice_sample_sd']:.8f}")
    print('Semantic rule supported   :',a['semantic_benefit_supported_by_predeclared_rule']); print('NEXT: Notebook 07')

if __name__=='__main__': main()
