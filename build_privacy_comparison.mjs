import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const root=process.cwd();
const skill='C:/Users/15397/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const runtime='C:/Users/15397/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(runtime+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const {finalizePresentation,applyPresentationChartFont,resolvePresentationFont}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const font=resolvePresentationFont({fontFamily:'Arial'});
const data=JSON.parse(await fs.readFile(path.join(root,'results/privacy_comparison_20260924/metrics.json'),'utf8'));
const matched=JSON.parse(await fs.readFile(path.join(root,'results/privacy_comparison_20260924/matched_budget.json'),'utf8'));
if(!Object.values(data.checks).every(Boolean)) throw Error('Benchmark checks failed');
const build=path.join(root,'.privacy-deck-build');
const output=path.join(root,'outputs');
await fs.mkdir(build,{recursive:true});await fs.mkdir(output,{recursive:true});
const p=Presentation.create({slideSize:{width:1280,height:720}});
const c={ink:'#173B36',muted:'#536B66',green:'#168276',orange:'#C46C32',blue:'#3E6B96',gray:'#7B8487',paper:'#F8FAF8',white:'#FFFFFF'};
const names=['BNP','Gaussian','Encryption','Encryption + Gaussian'];
const labels=['BNP','Gaussian','Encryption','Encryption +\nGaussian'];
const colors=[c.orange,c.blue,c.gray,c.green];
const notesBase='New measurements: results/privacy_comparison_20260924/metrics.json and trials.csv. Reproduce with run_privacy_comparison.py. Synthetic IEEE 123 data, event-level bounded replacement adjacency, ideal continuous-noise accounting. This is direct bus-sum release, separate from the existing model-fitting pipeline. All results use the same public input clipping. No topology or whole-customer trajectory privacy claim.';
function text(slide,str,x,y,w,h,size=25,color=c.ink,bold=false){
 const t=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=str;t.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none',insets:{top:0,right:0,bottom:0,left:0}};return t;
}
function slide(title,sub=''){
 const s=p.slides.add();s.background.fill=c.paper;
 text(s,title,64,45,1152,100,43,c.ink,true);
 if(sub)text(s,sub,64,145,1140,64,24,c.muted);
 text(s,String(p.slides.items.length).padStart(2,'0'),1170,670,50,24,17,c.muted);
 s.speakerNotes.textFrame.setText(notesBase);return s;
}
function note(s,str){s.speakerNotes.textFrame.setText(notesBase+'\n\n'+str);}
function bars(s,values,{x=64,y=240,w=1130,h=330,title='',format='0.0',max,categories=labels,seriesName='Measured value',palette=colors}={}){
 values=values.map(value=>Number(value.toPrecision(8)));
 const chart=s.charts.add('bar',{
  position:{left:x,top:y,width:w,height:h},categories,
  series:[{name:seriesName,values,valuesFormatCode:format,fill:palette[0],points:values.map((_,idx)=>({idx,fill:palette[idx%palette.length]}))}],
  barOptions:{direction:'column',grouping:'clustered',gapWidth:95},hasLegend:false,
  xAxis:{textStyle:{fontSize:21,fill:c.ink},majorGridlines:null},
  yAxis:{min:0,...(max?{max}:{}),title:{text:title,textStyle:{fontSize:20,fill:c.muted}},numberFormatCode:format,textStyle:{fontSize:19,fill:c.muted},majorGridlines:{fill:'#DAE3DF',width:1}},
  dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:21,bold:true,fill:c.ink}},
  chartFill:c.paper,plotAreaFill:c.paper,
 });applyPresentationChartFont(chart,{fontFamily:font});return chart;
}
function table(s,values){
 const tbl=s.tables.add({rows:values.length,columns:values[0].length,left:64,top:242,width:1152,height:308,columnWidths:[350,250,240,312],values});
 tbl.borders.assign({fill:'#DFE8E2',width:1,style:'solid'});
 for(let r=0;r<values.length;r++)for(let col=0;col<values[0].length;col++){
   const cell=tbl.getCell(r,col);cell.fill=r===0?c.ink:(r%2?c.white:'#EFF4F0');
   cell.text.style={typeface:font,fontSize:23,bold:r===0,color:r===0?c.white:c.ink};
 }
 return tbl;
}
const u=(n,k)=>data.utility[n][k].mean;
const fmt=(n,d=2)=>n.toFixed(d);
{
 const s=p.slides.add();s.background.fill=c.ink;
 text(s,'Privacy in voltage data',68,105,1130,90,59,c.white,true);
 text(s,'BNP, Gaussian noise and encryption',68,215,1100,100,40,'#A4D4C7');
 text(s,'A measured comparison on the IEEE 123 feeder',68,362,1060,55,30,c.white);
 text(s,'Encryption plus Gaussian noise adds collector confidentiality\nwhile preserving the Gaussian output distribution.',68,448,1080,110,30,c.white);
 text(s,'24 September 2026',68,636,720,30,20,'#A4D4C7');
 note(s,'Nine-slide explanation of a new direct bus-sum benchmark including an equal-budget test. Results do not replace prior model-fit experiments.');
}
{
 const s=slide('How Paillier encryption works','The collector can combine encrypted readings without opening them.');
 const steps=[['1  Meter','Adds its noise share, if enabled, then encrypts with the public key.'],['2  Collector','Combines ciphertexts to form an encrypted sum. It holds no private key.'],['3  Decryptor','Uses the private key to open the sum. A separate role receives only totals.']];
 steps.forEach(([a,b],i)=>{text(s,a,64,247+i*96,270,52,30,c.green,true);text(s,b,340,245+i*96,870,75,27);});
 text(s,'Example: encrypted 2 + encrypted 3 gives an encrypted total of 5',64,556,1140,52,27,c.ink,true);
 text(s,'Our simulation shares one process. The private-key holder could decrypt an individual ciphertext if it received one.',64,615,1120,55,22,c.muted);
 note(s,'Source: https://python-paillier.readthedocs.io/en/latest/phe.html and dpvolt/secure_agg.py. Paillier ciphertext multiplication implements plaintext addition modulo n, with fixed-point encoding for floats. Randomized 2048-bit encryption. Independent Gaussian noise shares have SD sigma/sqrt(10). Ten honest contributions yield variance sigma squared. Plain Paillier does not enforce aggregate-only decryption. Fixed rosters, authenticated messages and a separate decryptor are deployment requirements.');
}
{
 const s=slide('A shared comparison setup',`${data.settings.load_rows} load rows, 10 meters each, ${data.settings.steps} time samples and ${data.settings.trials} noise trials`);
 table(s,[['Method','Output ε','Output δ','Collector sees readings'],['BNP','0','0.01','Yes'],['Gaussian','1','0.01','Yes'],['Encryption','No output DP','No output DP','No*'],['Encryption + Gaussian','1','0.01','No*']]);
 text(s,'Privacy unit: replace one meter reading in public [0, 20] kW bounds.',64,578,1140,40,26,c.ink,true);
 text(s,'BNP uses a stricter epsilon at the same delta. *Assumes a separate, honest decryptor. Synthetic readings, no dropout.',64,628,1100,52,21,c.muted);
 note(s,`Public 20 kW input cap, sensitivity 20 kW. Input clipping affects ${(100*data.settings.input_clipped_fraction).toFixed(3)}% of meter readings. Its separate bus-load RMSE is ${data.settings.input_clipping_load_rmse_kw} kW. Multiple time samples retain this event-level guarantee because a neighboring dataset changes only one coordinate. Customer-trajectory adjacency needs new calibration or composition. The trusted BNP/Gaussian collector receives plaintext; these modes are not local DP.`);
}
{
 const s=slide('Privacy after the sum is revealed','Neighboring-sum attack accuracy. Closer to 50% means less distinguishable.');
 bars(s,names.map(n=>data.privacy[n].attack_empirical),{title:'Correct guesses',format:'0.0%',max:1.12});
 text(s,'Encryption alone reveals the exact sum to its recipient.',64,590,1140,39,29,c.ink,true);
 text(s,'Noisy methods: 200,000 decisions. Encryption: exact-sum limit. Chance = 50%. Raw outputs, before clipping.',64,636,1120,50,21,c.muted);
 note(s,'Equal prior labels for two neighboring datasets whose sums differ by the sensitivity, 20 kW. For Gaussian, the likelihood-ratio classifier thresholds at half the shift. For uniform BNP, classify disjoint support exactly and guess randomly in the overlap. Theory: Gaussian Phi(S/(2 sigma)), BNP 0.5+delta/2, exact plaintext sum 1. Paired Gaussian/hybrid use the same attack samples because their ideal output distribution matches. Encryption 100% is analytical, not a cryptanalytic attack. Each empirical noisy accuracy has approximate binomial 95% uncertainty of +/-0.22 percentage points. Source for Gaussian calibration: https://proceedings.mlr.press/v80/balle18a.html.');
}
{
 const s=slide('Load accuracy before output clipping','Average bus-sum RMSE across 30 trials. Lower is better.');
 bars(s,names.map(n=>u(n,'raw_load_rmse_kw')),{title:'RMSE (kW)',format:'0.0'});
 text(s,`BNP uses ±${fmt(data.settings.bnp_half_width_kw,0)} kW noise at δ = 0.01.`,64,590,1140,42,29,c.orange,true);
 text(s,`Gaussian noise SD = ${fmt(data.settings.gaussian_sd_kw)} kW. Encryption changes a sum by less than 0.000001 kW in this run.`,64,638,1120,50,21,c.muted);
 note(s,'Error is relative to the same publicly clipped input sums. Mean trial RMSE, not a privacy score. Zero-height encryption bar rounds away fixed-point error. BNP noise half-width B=S/(2 delta); uniform RMSE B/sqrt(3). Gaussian analytic calibration uses epsilon=1 and delta=.01. Encryption and hybrid use actual cryptography on every load row in the first trial. Subsequent hybrid trials use equivalent plaintext aggregation of identical local shares. Full trial-level 95% confidence intervals are in metrics.json.');
}
{
 const s=slide('Voltage accuracy after power flow','Magnitude RMSE on retained feeder nodes. Lower is better.');
 bars(s,names.map(n=>u(n,'voltage_magnitude_rmse_pu')),{title:'Voltage RMSE (pu)',format:'0.0000'});
 text(s,`${fmt(u('BNP','output_clipped_fraction')*100,1)}% of BNP sums hit the public clipping limits.`,64,590,1135,42,29,c.orange,true);
 text(s,'All sums use [0, 200] kW clipping before power flow. Controls stay fixed. All tested snapshots converged.',64,637,1130,52,21,c.muted);
 note(s,'Voltage magnitudes follow OpenDSS AC power flow with reactive loads determined by original public power factors. Every experiment resets the feeder. Reference voltages use the bounded input sums. Reported metric is mean across trial-specific RMSEs over six time samples and retained nodes. Public output clipping is post-processing and is essential context for this utility chart. It creates saturation and removes the large raw-noise tails. This direct-sum experiment does not inherit the model-fitting paper topology guarantee.');
}
{
 const s=slide('Equal privacy budget changes the ranking','Both mechanisms use ε = 0 and δ = 0.01 on the same query.');
 bars(s,[u('BNP','raw_load_rmse_kw'),matched.utility.raw_load_rmse_kw.mean],{categories:['BNP, ε = 0','Gaussian, ε = 0'],palette:[c.orange,c.blue],title:'Raw load RMSE (kW)',format:'0.0'});
 text(s,'BNP has lower raw error at this identical budget.',64,590,1140,43,29,c.ink,true);
 text(s,`Both clip about 90% of sums. Voltage RMSE: BNP ${fmt(u('BNP','voltage_magnitude_rmse_pu'),4)} pu, Gaussian ${fmt(matched.utility.voltage_magnitude_rmse_pu.mean,4)} pu.`,64,638,1130,52,21,c.muted);
 note(s,'Source: matched_budget.json and trials_matched.csv. Additional 30 independent Gaussian noise trials on the original inputs. At epsilon=0, Gaussian delta is its total variation: 2 Phi(S/(2 sigma))-1. Inversion gives sigma=S/[2 Phi^-1((1+delta)/2)]. Thus theoretical RMSE is 797.8637 kW for Gaussian versus 577.3503 kW for BNP. Both have theoretical equal-prior neighboring-sum attack success 50.5%. The initial Gaussian epsilon=1 results purchase utility by relaxing epsilon. Clipped voltage-error confidence intervals overlap, so this slide does not claim a meaningful voltage-utility advantage for BNP.');
}
{
 const enc=data.cryptography['Encryption'],hyb=data.cryptography['Encryption + Gaussian'];
 const s=slide('Measured encryption cost','One complete run per encrypted method, including every load row and time sample');
 bars(s,[enc.wall_s,hyb.wall_s],{categories:['Encryption','Encryption + Gaussian'],palette:[c.gray,c.green],title:'Wall time (seconds)',format:'0.0',y:225,h:335});
 text(s,`${enc.timing.encrypted_values.toLocaleString()} ciphertexts per run with 2048-bit Paillier`,64,590,1150,42,29,c.ink,true);
 text(s,`Key generation adds ${fmt(data.key_generation_s)} s separately. One machine, sequential execution and no network delay.`,64,638,1120,50,21,c.muted);
 note(s,JSON.stringify(data.cryptography,null,2)+'\nSingle observation for each method. Do not infer a statistical speed difference between encryption and hybrid. Times include local summing/harness overhead and exclude power flow, key generation, networking and production role separation. No claim of production throughput.');
}
{
 const s=p.slides.add();s.background.fill=c.ink;
 text(s,'What this comparison shows',64,62,1150,90,48,c.white,true);
 text(s,'Encryption preserves accuracy',64,202,1140,56,34,'#A4D4C7',true);
 text(s,'It hides inputs from the collector under the stated key-separation assumptions.',64,267,1120,74,28,c.white);
 text(s,'Gaussian noise protects the released sum',64,368,1140,55,34,'#A4D4C7',true);
 text(s,'Encryption keeps the ideal noise distribution. At equal privacy, BNP has lower raw error. Gaussian ε = 1 trades weaker privacy for better accuracy.',64,433,1120,112,28,c.white);
 text(s,'Scope: synthetic data, one-reading privacy, ideal noise arithmetic. Whole trajectories, dropout and production security need further work.',64,583,1120,74,23,'#C5DCD4');
 note(s,'Conclusions apply to this operating point and this direct bus-sum query. They are not a universal ranking of bounded-noise and Gaussian mechanisms. The two mechanisms have different epsilon guarantees. Research simulations use NumPy RNG and finite-precision arithmetic rather than a certified deployment-grade DP sampler. Paillier is real, while the trust boundary is simulated.');
}

const draft=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(draft);
console.log('Draft exported');
const final=path.join(output,'privacy_comparison.pptx');
await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:final,
 pythonExecutable:runtime+'/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','3'],
 requiredNativeTableOwnerSlides:[3],requiredNativeChartOwnerSlides:[4,5,6,7,8],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log('Finalized',final);
const deck=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<deck.slides.items.length;i++){
 const img=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await img.arrayBuffer()));
 console.log('Rendered slide',i+1);
}
