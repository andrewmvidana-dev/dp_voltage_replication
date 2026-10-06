import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root='C:/Users/15397/Desktop/dp_voltage_replication';
const skill='C:/Users/15397/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const runtime='C:/Users/15397/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(runtime+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const {finalizePresentation,resolvePresentationFont}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const build=root+'/.threshold-deck-build',out=root+'/outputs';
await fs.mkdir(build,{recursive:true});await fs.mkdir(out,{recursive:true});
await fs.copyFile('C:/Users/15397/.codex/generated_images/01a0f2b0-109d-7fb0-b0df-f9c0dfcb49b2/exec-52ea642c-bc32-4ddd-a225-a94c8117a05a.png',build+'/grid-cover.png');
const font=resolvePresentationFont({fontFamily:'Arial'});
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:'#102C3C',teal:'#137D7C',muted:'#536974',paper:'#F7F8F5',white:'#FFFFFF',gold:'#E9B365'};
const paper='https://doi.org/10.1007/s00145-023-09452-8';
const security='https://github.com/tuneinsight/lattigo/security';
function txt(s,v,x,y,w,h,size=28,color=C.ink,bold=false){const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});t.text=v;t.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};return t;}
function page(title,subtitle){const s=p.slides.add();s.background.fill=C.paper;txt(s,title,64,45,1152,82,46,C.ink,true);if(subtitle)txt(s,subtitle,64,135,1145,70,26,C.muted);txt(s,String(p.slides.items.length),1160,666,50,28,20,C.muted);return s;}
function note(s,t){s.speakerNotes.textFrame.setText(t);}
function table(s,vals,widths,y=225,h=330){const t=s.tables.add({rows:vals.length,columns:vals[0].length,left:64,top:y,width:1152,height:h,columnWidths:widths,values:vals});t.borders.assign({fill:'#D8E0DD',width:1,style:'solid'});for(let r=0;r<vals.length;r++)for(let c=0;c<vals[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?C.ink:(r%2?C.white:'#EDF3F0');cell.text.style={typeface:font,fontSize:25,bold:r===0,color:r===0?C.white:C.ink};}return t;}
{
 const s=p.slides.add();s.images.add({blob:new Uint8Array(await fs.readFile(build+'/grid-cover.png')),contentType:'image/png',alt:'Conceptual illustration of a residential electricity network',position:{left:0,top:0,width:1280,height:720},fit:'cover'});
 txt(s,'Encryption + privacy\nfor synthetic grids',64,120,640,175,57,C.white,true);
 txt(s,'How threshold encryption works\nwith BNP and differential privacy',64,332,590,110,31,'#B9E1D9');
 txt(s,'Proposed research architecture\nEvidence and next measurements',64,550,570,78,25,C.white);
 note(s,'Selected paper: Christian Mouchet, Elliott Bertrand and Jean-Pierre Hubaux, An Efficient Threshold Access-Structure for RLWE-Based Multiparty Homomorphic Encryption, Journal of Cryptology, 2023. '+paper+'\nThreshold BGV integration and joint BNP protocol are proposed, not implemented or benchmarked here. Conceptual grid artwork created using the built-in ImageGen tool. Prompt: restrained editorial residential distribution grid on the right of a deep navy 16:9 canvas, empty left half, cream houses and teal electric connections, no text.');
}
{
 const s=page('How the protections work together','Encryption protects computation. Calibrated noise protects the released statistics.');
 const rows=[['1','Meters compute statistics','Each meter keeps its raw records locally.'],['2','Noise enters before decryption','A calibrated protocol protects the aggregate.'],['3','The server adds encrypted values','It cannot read the individual contributions.'],['4','Authorities authorize a release','A quorum combines partial decryptions.'],['5','The model generates voltages','Synthetic loads pass through AC power flow.']];
 rows.forEach((r,i)=>{let y=220+i*75;txt(s,r[0],68,y,55,52,36,C.teal,true);txt(s,r[1],145,y,1010,37,29,C.ink,true);txt(s,r[2],145,y+37,1010,34,24,C.muted);});
 txt(s,'The fitter repeats this process for the mean and then the covariance.',64,625,1120,36,24,C.teal,true);
 note(s,'Proposed architecture adapted to the existing two-round fitter in dpvolt/secure_agg.py and SECURE_AGGREGATION.md. Current code uses Gaussian shares and single-key Paillier. A threshold BGV backend and a calibrated distributed BNP sampler remain future work. Local noise shares are not automatically local DP. Authorities need authenticated rosters, transcript checks and release-budget controls to enforce aggregate-only access. With a fixed public network, downstream generation is post-processing for customer-data privacy, not an automatic topology-privacy proof. Sources: '+paper+' ; https://arxiv.org/abs/2605.02390 ; local SECURE_AGGREGATION.md.');
}
{
 const s=page('A simple example of bounded noise','Illustrative aggregate statistic, in arbitrary units. These values are not a privacy calibration.');
 txt(s,'100',70,243,320,105,80,C.ink,true);txt(s,'Exact aggregate\nkept hidden',70,365,330,90,30,C.muted);
 txt(s,'+1.2',474,243,320,105,80,C.teal,true);txt(s,'Hidden noise draw\ninside [−2, +2]',474,365,350,90,30,C.muted);
 txt(s,'101.2',875,243,340,105,80,C.ink,true);txt(s,'Authorized\nreleased statistic',875,365,335,90,30,C.muted);
 txt(s,'BNP bounds the perturbation. DP quantifies the privacy guarantee.',64,500,1152,62,29,C.ink,true);
 txt(s,'A suitable bounded mechanism can satisfy DP itself. Gaussian noise added on top removes the hard bound on total noise.',64,588,1110,82,25,C.muted);
 note(s,'Arithmetic illustration only: 100 + 1.2 = 101.2 and the illustrative support is [-2,2]. A bound B does not establish DP without query sensitivity, adjacency, distribution and composition accounting. No voltage-error or grid-safety bound follows from this example. Independent uniform shares sum to a non-uniform distribution, so the existing Gaussian share construction cannot simply substitute uniform draws. Adding independent Gaussian noise yields unbounded support. Bounded-support approximate DP example: Dagan and Kur, A bounded-noise mechanism for differential privacy, https://arxiv.org/abs/2012.03817 . This is not asserted to be the repository-specific BNP paper. Encryption protects the noisy statistic during computation and cannot replace privacy calibration.');
}
{
 const s=page('Shared authority with small measured overhead','Proposed configuration and published benchmark use different numbers of authorities.');
 table(s,[['Number','Meaning','Evidence'],['3 of 5','Three cooperate to decrypt.\nTwo may be offline.','Proposed setting'],['12.0 ms','One authority creates its\ndecryption share.','Paper benchmark'],['0.4 ms','Threshold preparation within\nthose 12 ms, about 3.3%.','Paper benchmark']],[230,600,322],220,320);
 txt(s,'Benchmark: 7 of 20 authorities, parameter set III, Ryzen 9 5900X.',64,566,1130,40,24,C.muted);
 txt(s,'This measures local computation, not total grid or network latency.',64,620,1130,40,25,C.teal,true);
 note(s,'Source: Mouchet, Bertrand, Hubaux 2023, Section 4.2, '+paper+'. The authors report 12.0 ms per-party decryption-share generation with N=20, t=7, parameter set III, of which Combine takes 0.4 ms. Calculation: 0.4/12*100=3.333 percent of the reported operation, not total-system overhead or a speedup over Paillier. CPU AMD Ryzen 9 5900X, 3.7 GHz. Proposed 3-of-5 is not measured here. Under scheme assumptions, two authorities cannot decrypt alone; three colluding authorities can. This setting tolerates two unavailable authorities if the remaining three complete the protocol. It does not establish tolerance to meter dropout, malicious shares, or unsafe retries. '+security);
}
{
 const d=JSON.parse(await fs.readFile(root+'/results/secure_agg_fit.json','utf8'));const fit=d.model_fit.secure_aggregation;const share=100*fit.timing_totals.encryption_s/fit.fit_wall_s;
 const s=page('Batching targets the current bottleneck','Recorded Paillier baseline: 91 load rows, 10 simulated meters per row, 4 time samples.');
 txt(s,share.toFixed(1)+'%',64,239,450,115,94,C.teal,true);
 txt(s,'of fitting time spent\non encryption',64,375,470,90,32,C.ink,true);
 txt(s,fit.timing_totals.encryption_s.toFixed(2)+' s encryption',620,247,590,65,41,C.ink,true);
 txt(s,fit.fit_wall_s.toFixed(2)+' s total fit',620,332,590,65,41,C.ink,true);
 txt(s,'4,752 values per meter\nfor a 96-sample fit',620,435,590,95,32,C.teal,true);
 txt(s,'Packed BGV could encrypt many values together. The speedup remains unmeasured.',64,580,1110,80,27,C.ink,true);
 note(s,'Sources: results/secure_agg_fit.json and results/secure_agg_report.md, recorded 2026-09-22. Exact total fit wall time '+fit.fit_wall_s+' seconds. Encryption '+fit.timing_totals.encryption_s+' seconds. Derived share '+share+' percent. These are existing Paillier measurements, not new runs or threshold BGV results. Serial execution excludes network transfer; two processes overlapped during part of measurement. Workload at T=96: T+T*(T+1)/2=4752 scalar values per meter, multiplied by 910 simulated meters gives 4,324,320 scalar encryptions. The count is arithmetic, not a measured 96-step runtime. BGV batching requires suitable plaintext modulus and encoding and changes ciphertext sizes. No predicted speedup or direct comparison to the paper’s decryption microbenchmark is claimed. BGV-compatible threshold construction: '+paper);
}
{
 const s=page('The measurements that will decide success','The best settings must balance privacy, availability and voltage quality.');
 table(s,[['Measure','What a good result means'],['Privacy: ε, δ and bound B','A proved budget and bounded perturbation\nunder the stated honest-survivor count.'],['Voltage quality','Small voltage error and few limit violations\nat matched privacy budgets.'],['Cost and dropout','Lower total time and communication.\nValid releases when enough parties remain.']],[360,792],220,330);
 txt(s,'Next comparison: current Paillier versus packed threshold BGV.',64,578,1120,40,27,C.teal,true);
 txt(s,'BNP sampling, collusion protection and safe recovery still need validation.',64,631,1115,36,24,C.muted);
 note(s,'Proposed evaluation, not achieved results. Report epsilon/delta under a named adjacency (customer-day or full trajectory), compose both model-fitting rounds and repeated releases, and account for quantization/overflow and any failure probability. B bounds perturbation of the chosen statistic, not automatically voltage error. Evaluate minimum honest contributing meters separately from available decryption authorities. Compare BNP and Gaussian releases at matched guarantees before judging utility. Record voltage magnitude RMSE in pu, ANSI exceedance fraction, temporal correlation, convergence, total wall time, communication bytes, packing rate, and dropout bias. Lattigo current security guidance states multiparty retries may reveal key information and current retry countermeasures are not implemented: '+security+'. New backend remains a research prototype until protocol security is established. No first-in-CPS novelty claim.');
}
const draft=build+'/candidate.pptx',final=out+'/threshold_encryption_bnp_dp_explainer.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);console.log('Draft exported');
await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:final,pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','4','--require-native-table-slide','6'],requiredNativeTableOwnerSlides:[4,6],fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:build+'/validation.json'});
console.log('Finalized',final);
const deck=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<deck.slides.items.length;i++){const img=await deck.export({slide:deck.slides.items[i],format:'png',scale:1});await fs.writeFile(build+`/slide-${i+1}.png`,new Uint8Array(await img.arrayBuffer()));console.log('Rendered',i+1);}

