#!/usr/bin/env python3
"""Aggregate and plot OOD algorithm-identification sweeps."""
import argparse,csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.stats import t as student_t
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,default=HERE/'artifacts'/'summaries'/'ood_results.csv'); p.add_argument('--summary',type=Path,default=HERE/'artifacts'/'summaries'/'ood_summary.csv'); p.add_argument('--output-dir',type=Path,default=HERE/'figures'); args=p.parse_args()
 rows=list(csv.DictReader(args.input.open())); grouped=defaultdict(list)
 for r in rows: grouped[(r['family'],float(r['value']),r['method'])].append(float(r['mse']))
 summary=[]
 for (family,value,method),vals in sorted(grouped.items()):
  a=np.asarray(vals); se=a.std(ddof=1)/np.sqrt(len(a)); critical=student_t.ppf(.975,len(a)-1); summary.append({'family':family,'value':value,'method':method,'mean_mse':a.mean(),'stderr':se,'ci95_low':max(a.mean()-critical*se,0),'ci95_high':a.mean()+critical*se,'num_pools':len(a)})
 with args.summary.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n'); w.writeheader(); w.writerows(summary)
 args.output_dir.mkdir(parents=True,exist_ok=True); labels={'transformer':'Transformer','assignment_oracle':'Assignment oracle','known_pool_uniform':'Known pool (uniform prior)','em_ridge_K2':'EM ridge (K=2)','em_ridge_K3':'EM ridge (K=3)','ridge_current':'Current-task ridge','ridge_history':'History ridge'}
 for family,xlabel in [('similarity','Coefficient cosine similarity'),('imbalance','Majority probability'),('hierarchy','Hierarchy dispersion')]:
  fig,ax=plt.subplots(figsize=(7.5,4.8)); methods=['transformer','assignment_oracle','known_pool_uniform','em_ridge_K2' if family!='hierarchy' else 'em_ridge_K3','ridge_current','ridge_history']
  for method in methods:
   points=sorted([r for r in summary if r['family']==family and r['method']==method],key=lambda r:r['value'])
   ax.errorbar([r['value'] for r in points],[r['mean_mse'] for r in points],yerr=[r['stderr'] for r in points],marker='o',capsize=3,label=labels[method])
  ax.set(xlabel=xlabel,ylabel='Final-task MSE'); ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(args.output_dir/f'{family}.png',dpi=200); plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.8)); values=sorted({int(r['value']) for r in summary if r['family']=='component_count'})
 methods=['transformer','assignment_oracle','known_pool_uniform','em_ridge_K1','em_ridge_K2','em_ridge_K3','em_ridge_K4']
 for method in methods:
  points=sorted([r for r in summary if r['family']=='component_count' and r['method']==method],key=lambda r:r['value'])
  ax.errorbar([r['value'] for r in points],[r['mean_mse'] for r in points],yerr=[r['stderr'] for r in points],marker='o',capsize=3,label=labels.get(method,method.replace('_',' ')))
 ax.set(xlabel='Evaluation component count',ylabel='Final-task MSE'); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(True,alpha=.3); ax.legend(ncol=2); fig.tight_layout(); fig.savefig(args.output_dir/'component_count.png',dpi=200); plt.close(fig); print(f'Wrote {len(summary)} summary rows and four figures')
if __name__=='__main__': main()
