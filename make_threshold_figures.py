"""Publication-friendly figures from measured results; no invented timings."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results/threshold_20261005'
OUT=ROOT/'outputs/threshold_figures'
OUT.mkdir(parents=True,exist_ok=True)
d=json.loads((DATA/'metrics.json').read_text())
assert d.get('complete')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
 'axes.spines.right':False,'figure.facecolor':'#F7F8F5','axes.facecolor':'#F7F8F5',
 'text.color':'#102C3C','axes.labelcolor':'#102C3C','savefig.facecolor':'#F7F8F5'})
colors=['#CAD6D8','#102C3C','#137D7C'];methods=['None','Gaussian DP','BNP']
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=200,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.6))
for ax,key,label in zip(axs,['raw_rmse_kw','voltage_rmse_pu'],['Raw load RMSE (kW)','Voltage magnitude RMSE (pu)']):
 means=np.array([d['utility'][m][key]['mean'] for m in methods])
 errors=np.array([d['utility'][m][key]['ci95'][1]-d['utility'][m][key]['mean'] for m in methods])
 ax.bar(['No noise','Gaussian DP','BNP'],means,yerr=errors,capsize=5,color=colors)
 ax.set_ylabel(label);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 ax.set_ylim(0,max(means+errors)*1.25)
 for i,v in enumerate(means):ax.text(i,v+max(means)*.08,f'{v:.3g}',ha='center',fontweight='bold')
axs[0].set_title('Noise cost before output clipping');axs[1].set_title('Grid error after public clipping')
fig.suptitle('Matched privacy: ε = 0, δ = 0.01 per meter reading',fontweight='bold',y=1.05)
fig.text(.02,-.06,'30 synthetic trials; bars show means and 95% confidence intervals. Encryption preserves these mechanisms\nup to checked encoding error. No-noise output has no DP guarantee. Voltage inputs clipped to [0,200] kW.',fontsize=10)
save(fig,'matched_privacy_utility')

fig,ax=plt.subplots(figsize=(10,5))
x=np.arange(3);w=.35
for offset,backend,c in [(-w/2,'Paillier','#102C3C'),(w/2,'Threshold BGV','#137D7C')]:
 vals=[d['crypto'][backend+' / '+m]['wall_s'] for m in methods]
 bars=ax.bar(x+offset,vals,w,label=backend,color=c)
 ax.bar_label(bars,fmt='%.1f s',padding=5)
ax.set_xticks(x,['Encryption only','Encryption + Gaussian DP','Encryption + BNP']);ax.set_ylabel('Full-feeder elapsed time (seconds)');ax.legend(frameon=False)
ax.set_ylim(0,max(v['wall_s'] for v in d['crypto'].values())*1.25);ax.set_title('Measured cost on the same 91-row, 6-time input',fontweight='bold')
fig.text(.07,-.04,'One real encrypted trial per condition. BGV includes 91 fresh key setups; Paillier setup is separate.\nSingle machine, serial encryption; no network. Different precision/security parameters, not a security-equivalent speedup.',fontsize=10)
save(fig,'encryption_runtime')

fig,axs=plt.subplots(1,2,figsize=(12,4.5))
rows=d['dropout'];x=[r['lost_percent'] for r in rows]
axs[0].plot(x,[r['signal_rmse_kw'] for r in rows],'-o',color='#137D7C');axs[0].set_ylabel('Missing-load RMSE (kW)');axs[0].set_title('Missing meters remove real load')
axs[1].plot(x,[r['uncompensated_delta'] for r in rows],'-o',label='No noise adjustment',color='#102C3C')
axs[1].plot(x,[r['compensated_delta'] for r in rows],'--',label='Survivor-adjusted noise',color='#137D7C');axs[1].legend(frameon=False);axs[1].set_ylabel('Effective δ at ε = 0');axs[1].set_title('Missing noise shares weaken privacy')
for ax in axs:ax.set_xlabel('Meters missing (%)');ax.grid(alpha=.2)
fig.text(.03,-.06,'Missing-load error uses synthetic meter data. Privacy curves are analytic, assuming independent honest survivors.\nRestoring noise variance cannot restore missing load. Authority dropout is separate: 3 of 5 must cooperate.',fontsize=10)
save(fig,'meter_dropout')

fig,ax=plt.subplots(figsize=(8,4.8))
for key,c in [('bnp_rmse_kw','#137D7C'),('gaussian_rmse_kw','#102C3C')]:
 ax.loglog([r['delta'] for r in d['delta_sweep']],[r[key] for r in d['delta_sweep']],'-o',label='BNP' if key.startswith('bnp') else 'Gaussian DP',color=c)
ax.set_xlabel('δ (smaller means a stricter bound)');ax.set_ylabel('Theoretical raw noise RMSE (kW)');ax.legend(frameon=False);ax.grid(alpha=.2,which='both');ax.set_title('Stricter privacy increases noise cost',fontweight='bold')
fig.text(.01,-.14,'Analytic curves at ε = 0, scalar sensitivity 20 kW. Logarithmic axes.\nThese are noise calculations, not new encrypted feeder measurements.',fontsize=10)
save(fig,'privacy_budget_sweep')
print(OUT)

