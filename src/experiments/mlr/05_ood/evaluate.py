#!/usr/bin/env python3
"""Evaluate candidate algorithms under controlled MLR distribution shifts."""

from __future__ import annotations

import argparse, csv, sys, time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent; REPO_ROOT=HERE.parents[3]
sys.path[:0]=[str(REPO_ROOT/"src"),str(HERE.parent),str(HERE.parent/"01_behavior")]
from baselines import known_pool_bayes, ridge_current, ridge_history, unpack_tokens  # noqa:E402
from common.results import build_manifest, write_json  # noqa:E402
from common.runtime import load_condition, normalized_pool, sample_with_pool  # noqa:E402


def atomic_csv(rows,path):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def similarity_pool(similarity, d, seed):
    g=torch.Generator().manual_seed(seed); first=torch.randn(d,generator=g); first/=first.norm()
    other=torch.randn(d,generator=g); other-=torch.dot(other,first)*first; other/=other.norm()
    second=similarity*first+np.sqrt(max(1-similarity**2,0))*other
    return torch.stack((first,second))


def hierarchical_pool(K,d,dispersion,seed):
    g=torch.Generator().manual_seed(seed); center=torch.randn(d,generator=g); center/=center.norm()
    pool=center+dispersion*torch.randn(K,d,generator=g); return pool/pool.norm(dim=1,keepdim=True)


def assignments_uniform(K,B,N,seed):
    return torch.randint(K,(B,N),generator=torch.Generator().manual_seed(seed))


def assignments_imbalanced(probability,B,N,seed):
    g=torch.Generator().manual_seed(seed); ids=(torch.rand(B,N,generator=g)>probability).long(); ids[:,-1]=1; return ids


def em_final(X,y,query,components,noise_std,regularization,seed,iterations=20,initializations=5):
    """Vectorized multi-start EM on completed tasks, predicting only the final task."""
    X_hist,y_hist=X[:,:-1],y[:,:-1]; B,n,T,d=X_hist.shape; variance=max(noise_std**2,1e-6)
    grams_task=np.einsum('bnti,bntj->bnij',X_hist,X_hist); moments_task=np.einsum('bnti,bnt->bni',X_hist,y_hist)
    rng=np.random.default_rng(seed); best_score=np.full(B,-np.inf); best=np.zeros((B,components,d)); best_priors=np.full((B,components),1/components); eye=np.eye(d)[None,None]
    for _ in range(initializations):
        beta=rng.normal(scale=.1,size=(B,components,d)); priors=np.full((B,components),1/components)
        for _ in range(iterations):
            pred=np.einsum('bnti,bki->bnkt',X_hist,beta); ll=-.5*np.square(y_hist[:,:,None,:]-pred).sum(-1)/variance
            ll+=np.log(priors[:,None,:]+1e-12)
            ll-=ll.max(-1,keepdims=True); weights=np.exp(ll); weights/=weights.sum(-1,keepdims=True)
            priors=weights.mean(1)
            grams=np.einsum('bnk,bnij->bkij',weights,grams_task); moments=np.einsum('bnk,bni->bki',weights,moments_task)
            beta=np.linalg.solve(grams+regularization*eye,moments[...,None])[...,0]
        pred=np.einsum('bnti,bki->bnkt',X_hist,beta); ll=-.5*np.square(y_hist[:,:,None,:]-pred).sum(-1)/variance; ll+=np.log(priors[:,None,:]+1e-12)
        maximum=ll.max(-1,keepdims=True); score=(maximum[...,0]+np.log(np.exp(ll-maximum).sum(-1))).sum(-1)
        improved=score>best_score; best_score[improved]=score[improved]; best[improved]=beta[improved]; best_priors[improved]=priors[improved]
    pred=np.einsum('bti,bki->bkt',X[:,-1],best); ll=-.5*np.square(y[:,-1,None,:]-pred).sum(-1)/variance
    ll+=np.log(best_priors+1e-12)
    ll-=ll.max(-1,keepdims=True); weights=np.exp(ll); weights/=weights.sum(-1,keepdims=True)
    component_pred=np.einsum('bi,bki->bk',query[:,-1],best); return (weights*component_pred).sum(-1)


def evaluate_prompt(model,pool,assignments,*,T,N,noise,seed,device,em_components):
    B=len(assignments); xs,ys,_=sample_with_pool(pool=pool,T=T,N=N,noise_std=noise,batch_size=B,seed=seed,assignments=assignments)
    with torch.no_grad(): transformer=model(xs.to(device),ys.to(device)).cpu().numpy()
    X,y,query=unpack_tokens(xs.numpy().astype(np.float64),T,pool.shape[1]); target=ys.numpy().astype(np.float64)
    predictions={
        'transformer':transformer[:,-1],
        'known_pool_uniform':known_pool_bayes(X,y,query,pool.numpy().astype(np.float64),noise)[:,-1],
        'assignment_oracle':np.einsum('bi,bi->b',query[:,-1],pool.numpy()[assignments[:,-1].numpy()]),
        'ridge_current':ridge_current(X,y,query,regularization=max(noise**2*pool.shape[1],1e-6))[:,-1],
        'ridge_history':ridge_history(X,y,query)[:,-1],
    }
    for fit_K in em_components:
        predictions[f'em_ridge_K{fit_K}']=em_final(X,y,query,fit_K,noise,max(noise**2*pool.shape[1],.01),seed+fit_K*100)
    return {method:float(np.mean(np.square(pred-target[:,-1]))) for method,pred in predictions.items()}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--models-root',type=Path,required=True); p.add_argument('--output-dir',type=Path,default=HERE/'artifacts'/'summaries')
    p.add_argument('--families',nargs='+',choices=['similarity','imbalance','component_count','hierarchy'],default=['similarity','imbalance','component_count','hierarchy'])
    p.add_argument('--T',type=int,default=3); p.add_argument('--N',type=int,default=50); p.add_argument('--noise',type=float,default=.2)
    p.add_argument('--num-pools',type=int,default=10); p.add_argument('--num-prompts',type=int,default=128); p.add_argument('--seed',type=int,default=71_000); p.add_argument('--device',default='cpu'); args=p.parse_args()
    device=torch.device(args.device); model,_config,record=load_condition(args.models_root,T=args.T,K=2,N=args.N,noise_std=args.noise,device=device)
    conditions=[]
    if 'similarity' in args.families: conditions += [('similarity',v) for v in (-1.0,-.5,0,.5,.9,.99,1.0)]
    if 'imbalance' in args.families: conditions += [('imbalance',v) for v in (.5,.8,.95,.99)]
    if 'component_count' in args.families: conditions += [('component_count',v) for v in (1,2,3,4)]
    if 'hierarchy' in args.families: conditions += [('hierarchy',v) for v in (.05,.2,.5,1.0)]
    rows=[]; args.output_dir.mkdir(parents=True,exist_ok=True); csv_path=args.output_dir/'ood_results.csv'; started=time.monotonic()
    family_offsets={'similarity':0,'imbalance':1_000_000,'component_count':2_000_000,'hierarchy':3_000_000}
    for condition_index,(family,value) in enumerate(conditions):
        condition_started=time.monotonic()
        for pool_index in range(args.num_pools):
            # Pair sweep points by reusing the same base orientation, assignments,
            # and prompt noise within each family and pool.
            pool_seed=args.seed+family_offsets[family]+10_000+pool_index; prompt_seed=args.seed+family_offsets[family]+pool_index
            if family=='similarity': pool=similarity_pool(float(value),4,pool_seed); assignments=assignments_uniform(2,args.num_prompts,args.N,prompt_seed); em_K=(2,)
            elif family=='imbalance': pool=normalized_pool(2,4,pool_seed); assignments=assignments_imbalanced(float(value),args.num_prompts,args.N,prompt_seed); em_K=(2,)
            elif family=='component_count':
                K=int(value); pool=normalized_pool(K,4,pool_seed); assignments=assignments_uniform(K,args.num_prompts,args.N,prompt_seed); em_K=(1,2,3,4)
            else:
                pool=hierarchical_pool(3,4,float(value),pool_seed); assignments=assignments_uniform(3,args.num_prompts,args.N,prompt_seed); em_K=(3,)
            result=evaluate_prompt(model,pool,assignments,T=args.T,N=args.N,noise=args.noise,seed=prompt_seed+500,device=device,em_components=em_K)
            cosine=float(torch.dot(pool[0],pool[1])) if len(pool)>1 else 1.0
            minority_count=float((assignments[:,:-1]==1).sum(1).float().mean()) if family=='imbalance' else ''
            for method,mse in result.items(): rows.append({'family':family,'value':value,'pool_index':pool_index,'method':method,'mse':mse,'actual_pool_cosine':cosine,'mean_minority_context':minority_count,'num_prompts':args.num_prompts})
        atomic_csv(rows,csv_path); elapsed=time.monotonic()-started; done=condition_index+1; remaining=elapsed/done*(len(conditions)-done)
        print(f'{family}={value}: {time.monotonic()-condition_started:.1f}s; estimated remaining {remaining/60:.1f} min',flush=True)
    manifest_path=args.output_dir/'run_manifest.json'; write_json(build_manifest(repo_root=REPO_ROOT,experiment='05_ood_algorithm_identification',command=sys.argv,parameters=vars(args)|{'models_root':str(args.models_root),'output_dir':str(args.output_dir)},checkpoints=[asdict(record)],seeds={'base_seed':args.seed,'condition_stride':100_000,'pool_offset':10_000},outputs=[str(csv_path),str(manifest_path)]),manifest_path); print(f'Wrote {len(rows)} rows')

if __name__=='__main__': main()
