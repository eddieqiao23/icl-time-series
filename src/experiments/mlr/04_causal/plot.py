#!/usr/bin/env python3
"""Plot pool-level causal intervention summaries."""

import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent


def read(path):
    with path.open() as handle: return list(csv.DictReader(handle))


def mean_se(values):
    a=np.asarray(values,float); return a.mean(), a.std(ddof=1)/np.sqrt(len(a))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--summary-dir",type=Path,default=HERE/"artifacts"/"summaries")
    parser.add_argument("--output-dir",type=Path,default=HERE/"figures"); args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    heads=read(args.summary_dir/"head_ablation.csv")
    layers=sorted({int(r['layer']) for r in heads}); head_ids=sorted({int(r['head']) for r in heads})
    matrix=np.zeros((len(layers),len(head_ids)))
    for layer in layers:
        for head in head_ids:
            matrix[layer,head]=np.mean([float(r['delta_mse']) for r in heads if int(r['layer'])==layer and int(r['head'])==head])
    fig,ax=plt.subplots(figsize=(7,4.5)); bound=max(abs(matrix.min()),abs(matrix.max())); im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound)
    ax.set(xlabel='Head',ylabel='Layer',title='Head ablation effect'); ax.set_xticks(head_ids); ax.set_yticks(layers); fig.colorbar(im,ax=ax,label='Δ MSE'); fig.tight_layout(); fig.savefig(args.output_dir/'head_ablation.png',dpi=200); plt.close(fig)

    context=read(args.summary_dir/"context_ablation.csv"); fig,ax=plt.subplots(figsize=(7,4.5))
    labels={'same':'Same component','different':'Different component','random_count':'Random, count matched','all':'All context'}
    for condition in ('same','different','random_count','all'):
        positions=sorted({int(r['position']) for r in context}); means=[]; errors=[]
        for pos in positions:
            values=[float(r['delta_mse']) for r in context if r['condition']==condition and int(r['position'])==pos]; m,se=mean_se(values); means.append(m); errors.append(se)
        ax.errorbar(positions,means,yerr=errors,marker='o',capsize=3,label=labels[condition])
    ax.axhline(0,color='black',linewidth=.8); ax.set(xlabel='Task position',ylabel='Δ MSE'); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(args.output_dir/'context_ablation.png',dpi=200); plt.close(fig)

    support=read(args.summary_dir/"support_ablation.csv"); conditions=[f"support_{i}" for i in range(3)]+["all_supports"]
    means=[]; errors=[]
    for condition in conditions:
        values=[float(r['delta_mse']) for r in support if r['condition']==condition]; m,se=mean_se(values); means.append(m); errors.append(se)
    fig,ax=plt.subplots(figsize=(7,4.5)); x=np.arange(len(conditions)); ax.bar(x,means,yerr=errors,capsize=3)
    ax.set(xlabel='Replaced current-task support',ylabel='Δ MSE'); ax.set_xticks(x,['1','2','3','All']); ax.axhline(0,color='black',linewidth=.8); ax.grid(True,axis='y',alpha=.3); fig.tight_layout(); fig.savefig(args.output_dir/'support_ablation.png',dpi=200); plt.close(fig)

    patch=read(args.summary_dir/"activation_patching.csv"); fig,ax=plt.subplots(figsize=(7,4.5)); means=[]; errors=[]
    stages=sorted({int(r['stage']) for r in patch})
    for stage in stages:
        values=[float(r['recovery']) for r in patch if int(r['stage'])==stage]; m,se=mean_se(values); means.append(m); errors.append(se)
    ax.errorbar(stages,means,yerr=errors,marker='o',capsize=3); ax.axhline(0,color='black',linewidth=.8); ax.axhline(1,color='black',linewidth=.8,linestyle='--'); ax.set(xlabel='Patched representation stage',ylabel='Fraction of clean–corrupt gap recovered'); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(True,alpha=.3); fig.tight_layout(); fig.savefig(args.output_dir/'activation_patching.png',dpi=200); plt.close(fig)
    print('Wrote four causal figures')

if __name__=='__main__': main()
