import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const root='C:/Users/15397/Desktop/dp_voltage_replication';
const skill='C:/Users/15397/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const runtime='C:/Users/15397/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';
const {PresentationFile,FileBlob}=await import(pathToFileURL(runtime+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const build=root+'/.threshold-results-build';await fs.mkdir(build,{recursive:true});
const d=JSON.parse(await fs.readFile(root+'/results/threshold_20261005/metrics.json','utf8'));
if(!d.complete)throw Error('Experiments must finish first');
const p=await PresentationFile.importPptx(await FileBlob.load(root+'/outputs/threshold_encryption_bnp_dp_with_graphs.pptx'));
const inspected=await p.inspect({kind:'slide,textbox',maxChars:50000});await fs.writeFile(build+'/before.ndjson',inspected.ndjson);
const rec=inspected.ndjson.split('\n').filter(Boolean).map(JSON.parse);
for(const slide of [...p.slides.items].slice(2))slide.delete();
for(const item of rec.filter(x=>x.kind==='textbox'&&x.slide===1)){
 if(item.text.includes('Proposed research'))p.resolve(item.id).text='2023 threshold protocol implemented\nSynthetic-grid comparison results';
}
const C={ink:'#102C3C',teal:'#137D7C',muted:'#536974',paper:'#F7F8F5',white:'#FFFFFF',gold:'#E9B365',gray:'#CAD6D8'};
const font='Arial',paper='https://doi.org/10.1007/s00145-023-09452-8',nist='https://csrc.nist.gov/pubs/sp/800/226/final';
const security='https://github.com/tuneinsight/lattigo/security';
const source='Measured data: results/threshold_20261005/metrics.json. Reproduce with run_threshold_comparison.py. ';
function txt(s,v,x,y,w,h,size=28,color=C.ink,bold=false){const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});t.text=v;t.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};return t;}
function head(s,title,sub,n){s.background.fill=C.paper;txt(s,title,64,45,1152,82,44,C.ink,true);txt(s,sub,64,135,1145,70,25,C.muted);txt(s,String(n),1160,666,50,28,20,C.muted);}
function page(title,sub){const s=p.slides.add();head(s,title,sub,p.slides.items.length);return s;}
function note(s,t){s.speakerNotes.textFrame.setText(t);}
function table(s,vals,widths,y=222,h=332,size=24){const t=s.tables.add({rows:vals.length,columns:vals[0].length,left:64,top:y,width:1152,height:h,columnWidths:widths,values:vals});t.borders.assign({fill:'#D8E0DD',width:1,style:'solid'});for(let r=0;r<vals.length;r++)for(let c=0;c<vals[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?C.ink:(r%2?C.white:'#EDF3F0');cell.text.style={typeface:font,fontSize:size,bold:r===0,color:r===0?C.white:C.ink};}return t;}
function bar(s,categories,series,ytitle,opts={}){
 const ch=s.charts.add('bar',{position:{left:64,top:218,width:opts.width??1152,height:350},categories,
 series:series.map(a=>({...a,values:a.values.map(v=>Number(v.toPrecision(7))),valuesFormatCode:opts.format??'0.0'})),
 barOptions:{direction:'column',grouping:'clustered',gapWidth:95},hasLegend:series.length>1,
 legend:{position:'bottom',textStyle:{fontSize:20,fill:C.ink}},
 xAxis:{textStyle:{fontSize:22,fill:C.ink},majorGridlines:null},
 yAxis:{min:0,...(opts.max?{max:opts.max}:{}),title:{text:ytitle,textStyle:{fontSize:21,fill:C.muted}},textStyle:{fontSize:20,fill:C.muted},numberFormatCode:opts.format??'0.0',majorGridlines:{fill:'#D8E0DD',width:1}},
 dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:22,bold:true,fill:C.ink}},chartFill:C.paper,plotAreaFill:C.paper});applyPresentationChartFont(ch,{fontFamily:font});return ch;
}
note(p.slides.items[0],'Selected 2023 Journal of Cryptology paper by Christian Mouchet, Elliott Bertrand and Jean-Pierre Hubaux: '+paper+'. Actual Lattigo v6.2.0 research implementation, simulated colocated roles, not a production security deployment. Cover artwork retained from prior deck (ImageGen conceptual residential grid).');
const workflow=p.slides.items[1];
for(const item of rec.filter(x=>x.kind==='textbox'&&x.slide===2)){
 if(item.text==='Each meter keeps its raw records locally.')p.resolve(item.id).text='The prototype simulates local meter roles on one machine.';
 if(item.text==='It cannot read the individual contributions.')p.resolve(item.id).text='Its cryptographic interface uses the public key only.';
 if(item.text==='Authorities authorize a release')p.resolve(item.id).text='Three authorities cooperate to decrypt';
 if(item.text.startsWith('The fitter repeats'))p.resolve(item.id).text='Gaussian fitting works now. BNP testing uses direct sums and a trusted noise service.';
}
note(workflow,'Implemented Gaussian fitter: dpvolt/secure_agg.py fit_class, new dpvolt/threshold_agg.py and threshold_mhe/main.go. Noise enters before encryption. Threshold shares never reconstruct a complete key. Application authentication/authorization remains future work. BNP is implemented only for direct sums, using an independent trusted uniform-noise service. Fixed-grid power flow is customer-DP post-processing, not topology privacy. '+paper);
{
 const s=page('What the 2023 paper adds','Thresholdize creates secret shares. Combine prepares any committed quorum to decrypt.');
 table(s,[['Property','Previous Paillier','New research backend'],['Decryption authority','One complete private key','Any 3 of 5 authorities'],['Encrypted arithmetic','One scalar per ciphertext','Packed BGV vectors'],['Authority availability','Key holder must be available','Two may be absent at start'],['Released-output privacy','Depends on added noise','Still depends on added noise']],[330,355,467],220,330,24);
 txt(s,'The change is shared control and batching. The privacy budget stays separate.',64,588,1115,72,27,C.teal,true);
 note(s,paper+' ; https://pkg.go.dev/github.com/tuneinsight/lattigo/v6/multiparty . Paper published 2023, preprint 2022/780. Semi-honest setting. Implementation uses degree-two Shamir sharing of every authority component, public-key generation, Combine and collective key-switch-to-zero. No full secret-key reconstruction. All ten possible 3-of-5 quorums tested on fresh ciphertexts/keys. Mid-round dropout/retry is not supported. Do not infer 128-bit security from the parameter choices. '+security);
}
{
 const s=page('Encryption and output privacy','BNP can satisfy DP itself. Gaussian DP and BNP are alternative noise mechanisms here.');
 table(s,[['Noise mechanism','No encryption','Paillier','Threshold BGV'],['None','No output DP','No output DP','No output DP'],['Gaussian DP','(0, 0.01) ideal DP','Same ideal DP','Same ideal DP'],['BNP uniform noise','(0, 0.01) ideal DP','Same ideal DP','Same ideal DP']],[340,265,265,282],220,305,23);
 txt(s,'All nine conditions compared. Encryption changes access to inputs, not ε or δ.',64,559,1120,70,27,C.teal,true);
 txt(s,'BNP uses a trusted noise service. Finite-precision DP remains unproved.',64,630,1120,30,23,C.muted);
 note(s,source+'Privacy unit: replace one reading, one meter, one time. Public meter clipping [0,20] kW, sensitivity 20 kW. Uniform bound 1000 kW gives (0,.01) for ideal continuous scalar sum. Gaussian sigma 797.8637 kW matches that guarantee by total variation. These are illustrative stress settings, not standards-approved parameters. Encryption-only has no nontrivial output DP. The trusted BNP service can know exact aggregate by subtracting its noise and is excluded from the DP adversary. Gaussian assumes all reporting meters contribute independent hidden noise. '+nist);
}
{
 const s=page('Noise cost at the same privacy budget','One meter reading, ε = 0 and δ = 0.01. Lower raw RMSE means less distortion.');
 bar(s,['No noise','Gaussian DP','BNP'],[{name:'Raw RMSE',values:['None','Gaussian DP','BNP'].map(m=>d.utility[m].raw_rmse_kw.mean),fill:C.teal}], 'Raw load RMSE (kW)',{max:1000});
 const diff=d.paired_bnp_minus_gaussian.raw_rmse_kw;
 txt(s,`BNP minus Gaussian: ${diff.mean.toFixed(1)} kW. 95% CI [${diff.ci95[0].toFixed(1)}, ${diff.ci95[1].toFixed(1)}].`,64,590,1130,40,26,C.ink,true);
 txt(s,'30 noise trials. The encryption variants preserve this utility within encoding error.',64,635,1120,32,23,C.muted);
 note(s,source+'Bars show means of trial RMSE, not theoretical standard deviations. The displayed paired interval covers 30 seed-paired trial differences. Utility trials use plaintext-equivalent mechanism after checking real encryption for first trial of every condition. No-noise zero error has no output DP, so it is not a privacy competitor. BNP theoretical raw RMSE 577.3503 kW; Gaussian 797.8637 kW. Encryption does not reduce DP noise. All inputs synthetic; inference is conditional on this feeder.');
}
{
 const s=page('Voltage quality after clipping','Noisy loads are clipped to public [0,200] kW bounds before AC power flow.');
 bar(s,['No noise','Gaussian DP','BNP'],[{name:'Voltage RMSE',values:['None','Gaussian DP','BNP'].map(m=>d.utility[m].voltage_rmse_pu.mean),fill:C.ink}], 'Voltage magnitude RMSE (pu)',{format:'0.000',max:.13});
 txt(s,`Clipped outputs: Gaussian ${(100*d.utility['Gaussian DP'].clipped_fraction.mean).toFixed(1)}%; BNP ${(100*d.utility.BNP.clipped_fraction.mean).toFixed(1)}%.`,64,584,1130,40,27,C.teal,true);
 txt(s,'No clear voltage advantage: the paired 95% interval includes zero. Both errors are large.',64,630,1130,35,23,C.muted);
 note(s,source+'91 load rows, six time snapshots, 96 retained nodes. All measured power flows converged. RMSE averages voltage magnitudes against the publicly clipped noise-free reference. Convergence does not establish acceptable operational accuracy. 95% intervals: '+JSON.stringify({Gaussian:d.utility['Gaussian DP'].voltage_rmse_pu,BNP:d.utility.BNP.voltage_rmse_pu})+'. Paired BNP minus Gaussian interval: '+JSON.stringify(d.paired_bnp_minus_gaussian.voltage_rmse_pu)+'. Public clipping is ideal DP post-processing but changes utility. These are direct-sum experiments, distinct from synthetic-model fitting.');
}
{
 const s=page('Measured full-feeder encryption cost','Same 91 rows and six time samples. Each condition performs real cryptography once.');
 bar(s,['No noise','Gaussian DP','BNP'],['Paillier','Threshold BGV'].map((b,i)=>({name:b,values:['None','Gaussian DP','BNP'].map(m=>d.crypto[b+' / '+m].wall_s),fill:i?C.teal:C.ink})),'Elapsed seconds');
 txt(s,'BGV includes fresh setup for every bus. Network transfer is not measured.',64,590,1135,36,26,C.ink,true);
 txt(s,'These implementations use different precision and security parameters.',64,635,1120,32,24,C.muted);
 note(s,source+'Each bus has ten meter contributions and an extra noise-service contribution, zero outside BNP. Paillier: 6006 scalar ciphertexts, one key setup outside timed releases; key setup is separately recorded. BGV: 1001 packed ciphertexts across 91 fresh key epochs, setup and process overhead included in wall time. Thus this is observed implementation cost, not security-equivalent algorithm speedup. Serial workers=1. Windows single-machine run; no LAN overhead or production hardening. No timing confidence interval from this single run.');
}
{
 const s=page('Batching and parallel processing','Matched microbenchmark: ten meters, 96 values each, three fresh runs per backend.');
 bar(s,['Paillier','BGV serial','BGV 4 workers'],[{name:'Mean elapsed time',values:['Paillier','BGV serial','BGV 4 workers'].map(m=>d.batching[m].mean_s),fill:C.teal}],'Mean elapsed seconds',{width:815,format:'0.000'});
 txt(s,'960 vs 10',920,263,300,70,44,C.teal,true);txt(s,'ciphertexts for\nthe same input\nvector workload',920,364,300,132,28,C.ink);
 txt(s,'Packing reduces ciphertext count. Parallel speed depends on workload and overhead.',64,587,1130,74,26,C.ink,true);
 note(s,source+'Three raw elapsed measurements per condition: '+JSON.stringify(Object.fromEntries(Object.entries(d.batching).map(([k,v])=>[k,v.wall_s])))+'. Count reduction is scalar Paillier vs one packed BGV ciphertext per meter, not a 96-fold runtime claim. BGV supports 16384 slots but only 96 used in this benchmark. 4 worker goroutines parallelize independent encryptions; authority operations remain serial. Fresh threshold setup included, Paillier setup separate. Different precision and security settings. Serialized payload can grow even with fewer ciphertexts.');
}
{
 const s=page('The communication tradeoff','Short six-value vectors leave almost all BGV slots unused. Fewer ciphertexts can still mean more bytes.');
 const messages=d.crypto['Threshold BGV / None'].releases;
 const cb=messages.reduce((a,r)=>a+r.ciphertext_bytes,0)/1048576;
 const sb=messages.reduce((a,r)=>a+r.decryption_share_bytes,0)/1048576;
 bar(s,['Paillier integers\nfixed-width estimate','BGV ciphertexts\nserialized size','BGV decryption shares\nserialized size'],[{name:'Payload size',values:[6006*512/1048576,cb,sb],fill:C.teal}],'Payload size (MiB)',{max:600});
 txt(s,'BGV needs denser packing before network deployment. Setup traffic is extra.',64,590,1130,62,27,C.ink,true);
 note(s,source+'Computed serialized BGV ciphertext sizes and decryption-share sizes reported by Lattigo BinarySize, summed across91 releases. These are payload sizes, not network measurements. Paillier estimate6006 ciphertext integers times512bytes for fixed-width4096-bit integers, excludes exponent/key metadata. BGV excludes all setup/public-key/resharing traffic and network framing; the separate share bar is additional to ciphertexts. Thus the total BGV payload is at least about569MiB in this sparse six-slot workload, despite lower CPU time. Microbench96slots also leaves spare capacity. Packing longer mean/covariance vectors can use capacity better without combining unrelated households in plaintext.');
}
{
 const s=page('Authority dropout and meter dropout','Three decryption authorities are enough. Missing customer readings still change the result.');
 bar(s,d.dropout.filter(r=>r.lost_percent%20===0).map(r=>r.lost_percent+'%'),[{name:'Missing-load RMSE',values:d.dropout.filter(r=>r.lost_percent%20===0).map(r=>r.signal_rmse_kw),fill:C.ink}],'Missing-load RMSE (kW)',{width:800});
 txt(s,'3 of 5',914,253,304,75,57,C.teal,true);txt(s,'All 10 valid\nquorums tested\non fresh releases',914,359,304,120,27,C.ink);
 txt(s,'At 50% meter loss, unadjusted Gaussian δ rises from 0.0100 to '+d.dropout.find(r=>r.lost_percent===50).uncompensated_delta.toFixed(4)+'.',64,585,1130,72,26,C.ink,true);
 note(s,source+'Quorum correctness tested by verify_threshold.py. Fewer than three or duplicate/invalid identities refused. This is pre-committed authority availability, not a tested network recovery mechanism. Missing-load chart is calculated from surviving synthetic meters without noise; no imputation/rescaling. Delta curve analytic at epsilon0 assuming all survivors independent and honest: effective sigma=original sigma sqrt(k/10). Survivor compensation restores nominal variance, not signal. Missing profile data and dishonest meters require separate handling.');
}
{
 const s=page('The existing model fitter now uses BGV','The new backend runs both encrypted releases: class means and clipped covariance entries.');
 txt(s,d.model_fit.wall_s.toFixed(2)+' s',64,246,485,100,76,C.teal,true);
 txt(s,'Measured full-feeder\nGaussian model fit',64,372,495,105,32,C.ink,true);
 txt(s,'91 load rows, 910 simulated meters\n45 days, four time samples\nThree classes, six encrypted rounds',620,248,590,160,29,C.ink);
 txt(s,'Rounding error vs plaintext\nMean: '+d.model_fit.max_mean_error.toFixed(8)+'\nCovariance: '+d.model_fit.max_covariance_error.toFixed(8),620,433,590,120,27,C.teal,true);
 txt(s,'Separate fit test: ε = 50, δ = 0.00001. BNP model fitting is still future work.',64,587,1130,75,25,C.muted);
 note(s,source+'Fit settings epsilon=50,delta=1e-5,public synthetic load bounds [.0001,1]pu,T4,45days. This is an integration smoke test, not the eps0 direct-sum comparison. Errors measured in log-load mean and log-load covariance units. Paired plaintext reference uses same RNG to isolate quantization. Earlier Paillier result215.54s used a different archive, so no matched model-fit speedup is asserted. Data remain proportional simulated customers. BNP sum primitive is implemented, but joint BNP mean/covariance privacy accounting is not.');
}
{
 const s=page('Privacy guidance and remaining limits','NIST SP 800-226 guides the evaluation. It does not certify this prototype.');
 table(s,[['Requirement','Current evidence or limit'],['Privacy unit and budget','One reading per meter; ε = 0, δ = 0.01 in the comparison'],['Noise and trust','Honest Gaussian contributors; trusted BNP noise service'],['Repeated releases','Offline trials only; a deployment needs composition accounting'],['Cryptographic security','Real BGV operations; colocated roles and unaudited flooding'],['Finite precision','Encoding error tested; implemented DP proof still required']],[340,812],220,354,23);
 txt(s,'No claim of NIST compliance, certified 128-bit security or full-customer privacy.',64,603,1130,56,25,C.teal,true);
 note(s,nist+' ; '+security+' . NIST 2025 evaluation guidance motivates adjacency, bounded contribution, mechanism, trust model and accounting disclosures; no universal epsilon recommendation is inferred. This remains research. Experimental parameters LogN14, LogQ55+55,LogP55,plaintext1099511922689,scale1e6,flooding sigma2^40. No derived decryption leakage bound or independent lattice estimator assessment. Fresh epochs avoid exposed retry endpoints but do not prove full protocol security. Authenticated distributed services, malicious-share verification and persistent budget control remain absent.');
}
{
 const s=page('What this contributes to your research','A working threshold-encryption extension with measured costs and explicit privacy assumptions.');
 txt(s,'Implemented',64,239,520,50,35,C.teal,true);
 txt(s,'Packed threshold BGV\nGaussian private model fitting\nBNP encrypted direct sums\nMatched utility and dropout graphs',64,318,530,230,30,C.ink);
 txt(s,'Needed for a stronger paper',668,239,545,50,35,C.ink,true);
 txt(s,'Dealer-free bounded-noise protocol\nA finite-precision privacy proof\nIndependent customer profiles\nDistributed adversarial evaluation',668,318,550,230,29,C.ink);
 txt(s,'Novelty should rest on a new proved method or finding. “First to combine” is unverified.',64,595,1135,64,26,C.muted);
 note(s,'Selected foundation: '+paper+'. The implemented contribution is an integration and evaluation, not an independent cryptographic invention or established first-in-CPS claim. Prior literature already combines encryption and differential privacy in cyber-physical applications. Future BNP distributed sampling must justify exact joint distribution and collusion assumptions; adding arbitrary uniform shares is insufficient. Full-customer trajectories require adjacency and privacy composition beyond this event-level comparison.');
}
const draft=build+'/candidate.pptx',final=root+'/outputs/threshold_encryption_comparison.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);console.log('Draft exported');
await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:final,pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','3','--require-native-table-slide','4','--require-native-table-slide','12'],requiredNativeTableOwnerSlides:[3,4,12],requiredNativeChartOwnerSlides:[5,6,7,8,9,10],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:build+'/validation-comparison.json'});
const finished=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<finished.slides.items.length;i++){const png=await finished.export({slide:finished.slides.items[i],format:'png',scale:1});await fs.writeFile(build+`/slide-${i+1}.png`,new Uint8Array(await png.arrayBuffer()));console.log('Rendered',i+1);}
console.log(final);
