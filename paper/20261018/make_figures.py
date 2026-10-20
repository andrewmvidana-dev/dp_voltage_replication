"""Publication figures from immutable October 13 experiment artifacts.

Run with the repository virtual environment; no experiments are rerun.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE/'data' if (HERE/'data').exists() else ROOT/'results'/'threat_model_20261013'
OUT = HERE / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
def read(name):
    return json.loads((DATA/name).read_text(encoding='utf-8-sig'))
c,k,v,t = [read(x) for x in ['comparison.json','key_compromise.json','voltage_attack.json','timing.json']]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
    'axes.titlesize':9,'xtick.labelsize':7.3,'ytick.labelsize':7.3,'legend.fontsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
colors = {'Gaussian DP':'#1672A5','BNP':'#B76B17','Distributed BNP':'#23836B','None':'#626774'}
names = {'Gaussian DP':'Gaussian','BNP':'Central uniform','Distributed BNP':'Distributed uniform','None':'No noise'}
def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
def attack(m,observer):
    return next(x for x in k['cells'] if x['backend']=='Paillier' and x['mechanism']==m and
                x['adversary']==observer and x['channel']=='full_view' and x['target']=='target_meter')
def point(ax,mean,ci,y,**kwargs):
    ax.errorbar(mean,y,xerr=[[mean-ci[0]],[ci[1]-mean]],fmt='o',capsize=2.4,lw=1,ms=4,**kwargs)

# A: full leakage range. B: resolve the small advantages hidden by the full scale.
fig,axs = plt.subplots(1,2,figsize=(7.05,2.35),gridspec_kw={'width_ratios':[1,1.12]})
for j,m in enumerate(['None','BNP','Gaussian DP','Distributed BNP']):
    x=attack(m,'Key thief')['measured']; point(axs[0],100*x['accuracy'],np.array(x['ci95'])*100,j,color=colors[m])
    axs[0].text(100*x['accuracy']+(1 if x['accuracy']<.9 else -1),j-.14,f"{100*x['accuracy']:.3f}%",
                ha='left' if x['accuracy']<.9 else 'right',fontsize=7)
axs[0].set(yticks=range(4),yticklabels=[names[m] for m in ['None','BNP','Gaussian DP','Distributed BNP']],
    xlabel='Correct neighboring-world decisions (%)',xlim=(47,103),ylim=(3.45,-.5),title='(a) Full message transcript after key theft')
axs[0].axvline(50,color='#888888',ls=':',lw=.8)
mechanisms=['Gaussian DP','BNP','Distributed BNP']
for j,m in enumerate(mechanisms):
    a=attack(m,'Public analyst')['measured']; z=v['voltage_only_attacks'][m]
    point(axs[1],100*a['accuracy'],np.array(a['ci95'])*100,j-.15,color='#303D50',label='Public raw sum' if j==0 else None)
    axs[1].errorbar(100*z['accuracy'],j+.15,xerr=[[100*(z['accuracy']-z['ci95_wilson'][0])],[100*(z['ci95_wilson'][1]-z['accuracy'])]],
        fmt='s',color='#8A4383',capsize=2.4,ms=3.6,lw=1,label='Voltage-only AC inversion' if j==0 else None)
axs[1].axvline(50,color='#888888',ls=':',lw=.8,label='Chance: 50%')
axs[1].axvline(50.5,color='#D08423',ls='--',lw=.9,label='Ideal target: 50.5%')
axs[1].set(yticks=range(3),yticklabels=[names[m] for m in mechanisms],xlabel='Correct neighboring-world decisions (%)',
    xlim=(49.25,51.55),ylim=(2.65,-.6),title='(b) Public observations with noise (zoom)')
for ax in axs: ax.grid(axis='x',alpha=.15)
axs[1].legend(loc='upper center',bbox_to_anchor=(.48,-.23),ncol=2,frameon=False,fontsize=6.8)
fig.subplots_adjust(wspace=.55,bottom=.27,top=.86)
save(fig,'privacy_accuracy')

fig,axs=plt.subplots(2,1,figsize=(3.43,2.85))
for ax,field,label in [(axs[0],'raw_rmse_kw','Raw load RMSE (kW)'),(axs[1],'voltage_rmse_pu','Voltage-magnitude RMSE (pu)')]:
    for j,m in enumerate(mechanisms):
        x=c['utility'][m][field]
        ax.barh(j,x['mean'],color=colors[m],height=.55,alpha=.85)
        ax.errorbar(x['mean'],j,xerr=[[x['mean']-x['ci95'][0]],[x['ci95'][1]-x['mean']]],fmt='none',color='#222222',capsize=2,lw=.8)
    ax.set(yticks=range(3),yticklabels=[names[m] for m in mechanisms],xlabel=label,ylim=(2.5,-.5))
    ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
axs[0].set_xlim(0,860); axs[1].set_xlim(0,.106)
fig.subplots_adjust(left=.34,right=.99,hspace=.62,top=.96,bottom=.13)
save(fig,'utility_cost')

fig,axs=plt.subplots(1,2,figsize=(7.05,2.45),gridspec_kw={'width_ratios':[.95,1.15]})
for j,b in enumerate(['Paillier','Threshold BGV']):
    for offset,m in zip([-.24,-.08,.08,.24],['None']+mechanisms):
        x=t['summary'][b+' / '+m]
        axs[0].errorbar(x['mean'],j+offset,xerr=x['std_s'],fmt='o',color=colors[m],capsize=2,ms=4,label=names[m] if j==0 else None)
axs[0].set_xscale('log')
axs[0].set(xlabel='Wall time (s, logarithmic scale)',yticks=[0,1],yticklabels=['Paillier','Threshold BGV'],
    ylim=(1.6,-.6),xlim=(.17,30),title='(a) Fresh setup and aggregation')
axs[0].legend(loc='upper center',bbox_to_anchor=(.51,-.22),ncol=2,frameon=False,fontsize=6.8)
axs[1].axvspan(-10,10,color='#DCEDE4',alpha=.8,zorder=0)
axs[1].axvline(0,color='#777777',lw=.7); axs[1].axvline(-10,color='#70987C',ls=':',lw=.7); axs[1].axvline(10,color='#70987C',ls=':',lw=.7)
labels=[]
for j,(b,m) in enumerate([(b,m) for b in ['Paillier','Threshold BGV'] for m in mechanisms]):
    x=t['comparisons'][b+' / '+m]['relative_difference']
    point(axs[1],100*x['mean'],np.array(x['ci95'])*100,j,color=colors[m])
    labels.append(('P / ' if b=='Paillier' else 'B / ')+{'Gaussian DP':'Gaussian','BNP':'Central','Distributed BNP':'Distributed'}[m])
axs[1].set(yticks=range(6),yticklabels=labels,ylim=(5.6,-.7),xlim=(-12,17),xlabel='Paired runtime change from no noise (%)',title='(b) 95% CI and +/-10% margin')
for ax in axs: ax.grid(axis='x',alpha=.15)
fig.subplots_adjust(left=.13,right=.99,wspace=.55,top=.86,bottom=.27)
save(fig,'runtime_cost')

# Small validation figure, available separately and usable in future revisions.
fig,ax=plt.subplots(figsize=(3.43,2.2))
for j,m in enumerate(['None']+mechanisms):
    r=v['cases'][m]
    # Schema deliberately printed below rather than silently guessing metric names.
    ax.bar(j,c['utility'][m]['clipped_rmse_kw']['mean'],color=colors[m],alpha=.85)
ax.set(xticks=range(4),xticklabels=['None','Gaussian','Central','Distributed'],ylabel='Clipped load RMSE (kW)',ylim=(0,125))
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout()
save(fig,'clipped_load_error')
print('Created four data-derived figure sets (PDF/SVG/PNG).')
