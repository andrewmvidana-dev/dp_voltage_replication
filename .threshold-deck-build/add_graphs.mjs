import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const root='C:/Users/15397/Desktop/dp_voltage_replication';
const skill='C:/Users/15397/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const runtime='C:/Users/15397/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';
const {PresentationFile,FileBlob}=await import(pathToFileURL(runtime+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const build=root+'/.threshold-graphs-build';await fs.mkdir(build,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(root+'/outputs/threshold_encryption_bnp_dp_explainer.pptx'));
const snapshot=await p.inspect({kind:'slide,textbox,table',maxChars:30000});await fs.writeFile(build+'/before.ndjson',snapshot.ndjson);
const records=snapshot.ndjson.split('\n').filter(Boolean).map(v=>JSON.parse(v));
const slides=records.filter(v=>v.kind==='slide');
const old4=p.resolve(slides.find(v=>v.slide===4).id);
const old5=p.resolve(slides.find(v=>v.slide===5).id);
const old6=p.resolve(slides.find(v=>v.slide===6).id);
const s3=p.resolve(slides.find(v=>v.slide===3).id);
const C={ink:'#102C3C',teal:'#137D7C',muted:'#536974',paper:'#F7F8F5',white:'#FFFFFF',gold:'#E9B365',gray:'#CAD6D8'};
const font='Arial',paper='https://doi.org/10.1007/s00145-023-09452-8';
function txt(s,v,x,y,w,h,size=28,color=C.ink,bold=false){let t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});t.text=v;t.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};return t;}
function head(s,title,sub,n){s.background.fill=C.paper;txt(s,title,64,45,1152,82,46,C.ink,true);txt(s,sub,64,135,1145,70,26,C.muted);txt(s,String(n),1160,666,50,28,20,C.muted);}
const axis=(title)=>({min:0,title:{text:title,textStyle:{fontSize:22,fill:C.muted}},textStyle:{fontSize:22,fill:C.muted},majorGridlines:{fill:'#D8E0DD',width:1}});
old4.delete();
const {slide:s4}=p.slides.insert({after:s3});
head(s4,'Threshold preparation takes 0.4 ms','Published breakdown of one authority’s 12 ms decryption-share computation',4);
const ch4=s4.charts.add('bar',{position:{left:64,top:226,width:775,height:340},categories:['Other share\ncomputation','Threshold\npreparation'],series:[{name:'CPU time (ms)',values:[11.6,0.4],valuesFormatCode:'0.0',fill:C.ink,points:[{idx:0,fill:C.ink},{idx:1,fill:C.gold}]}],barOptions:{direction:'column',grouping:'clustered',gapWidth:125},hasLegend:false,xAxis:{textStyle:{fontSize:23,fill:C.ink},majorGridlines:null},yAxis:{...axis('CPU time (milliseconds)'),max:14},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:26,bold:true,fill:C.ink}},chartFill:C.paper,plotAreaFill:C.paper});applyPresentationChartFont(ch4,{fontFamily:font});
txt(s4,'3.3%',900,261,306,95,75,C.teal,true);txt(s4,'of this operation\nfor threshold\npreparation',900,365,306,130,28,C.ink);
txt(s4,'7 of 20 authorities, parameter set III, Ryzen 9 5900X. Network time excluded.',64,589,1130,62,24,C.muted);
s4.speakerNotes.textFrame.setText('Source: Mouchet, Bertrand and Hubaux 2023, Section 4.2, '+paper+'. Published total is 12.0 ms per party for a decryption share at N=20, t=7, parameter set III. Combine costs 0.4 ms. Other computation is derived as 12.0-0.4=11.6 ms; share is 0.4/12=3.333 percent. These are not separate independent observations or a speedup over Paillier. Published local CPU benchmark, no networking. Current Lattigo limitations on retries and decryption leakage still apply. Proposed project setup remains 3-of-5, allowing two unavailable authorities if the remaining three complete the protocol under its assumptions.');
old5.shapes.deleteAll();
head(old5,'Encryption uses 99.4% of fitting time','Measured Paillier baseline on the full feeder with four time samples',5);
const d=JSON.parse(await fs.readFile(root+'/results/secure_agg_fit.json','utf8'));const f=d.model_fit.secure_aggregation;const enc=f.timing_totals.encryption_s,total=f.fit_wall_s,other=total-enc;
const ch5=old5.charts.add('doughnut',{position:{left:65,top:210,width:640,height:350},categories:['Encryption','Other fitting time'],series:[{name:'Seconds',values:[Number(enc.toFixed(6)),Number(other.toFixed(6))],points:[{idx:0,fill:C.teal},{idx:1,fill:C.gold}]}],doughnutOptions:{holeSize:62,firstSliceAngle:270},hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:24,fill:C.ink}},dataLabels:{showValue:false,showPercent:false},chartFill:C.paper,plotAreaFill:C.paper});applyPresentationChartFont(ch5,{fontFamily:font});
txt(old5,enc.toFixed(2)+' s',775,233,440,86,65,C.teal,true);txt(old5,'Encryption',775,324,440,42,29,C.ink);
txt(old5,other.toFixed(2)+' s',775,398,440,72,52,C.ink,true);txt(old5,'All other fitting work',775,475,440,44,29,C.muted);
txt(old5,'215.54 s total. Batching targets almost all of the current fitting cost.',64,587,1140,67,28,C.ink,true);
old5.speakerNotes.textFrame.setText('Measured source: results/secure_agg_fit.json and results/secure_agg_report.md, 2026-09-22. Encryption '+enc+' s, total '+total+' s, other fitting time (derived residual) '+other+' s. Percentage '+(enc/total*100)+'%. The residual includes addition, decryption, key generation and harness/model overhead, not just decryption. 91 load rows, ten proportional simulated meters per row, four time coordinates, 12,740 scalar encryptions. Serial wall time, network excluded, two benchmark processes overlapped during part of measurement. This is Paillier and Gaussian fitting, not a BGV measurement. No new benchmark run was performed.');
const {slide:s6}=p.slides.insert({after:old5});
head(s6,'Longer profiles sharply increase the workload','Calculated scalar encryptions per meter in the current unbatched fitter',6);
const horizons=[4,24,48,96],counts=horizons.map(t=>t+t*(t+1)/2);
const ch6=s6.charts.add('bar',{position:{left:64,top:220,width:810,height:342},categories:horizons.map(t=>String(t)),series:[{name:'Values per meter',values:counts,valuesFormatCode:'#,##0',fill:C.teal,points:[{idx:3,fill:C.ink}]}],barOptions:{direction:'column',grouping:'clustered',gapWidth:95},hasLegend:false,xAxis:{title:{text:'Time samples in each profile',textStyle:{fontSize:22,fill:C.muted}},textStyle:{fontSize:23,fill:C.ink},majorGridlines:null},yAxis:{...axis('Scalar encryptions per meter'),max:5500,numberFormatCode:'#,##0'},dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:23,bold:true,fill:C.ink}},chartFill:C.paper,plotAreaFill:C.paper});applyPresentationChartFont(ch6,{fontFamily:font});
txt(s6,'4.32 million',914,268,305,116,51,C.teal,true);txt(s6,'scalar encryptions\nfor 910 meters\nat 96 samples',914,405,305,130,27,C.ink);
txt(s6,'Packed BGV can group values. Its actual speedup still needs measurement.',64,595,1130,65,27,C.ink,true);
s6.speakerNotes.textFrame.setText('Calculated from the implemented mean and covariance statistics: T + T(T+1)/2. Counts for T=[4,24,48,96] are [14,324,1224,4752]. Across 91 load rows with ten simulated meters per row: 4752*910=4,324,320 scalar encryptions at T=96. Source: SECURE_AGGREGATION.md, dpvolt/secure_agg.py and results/secure_agg_report.md. Counts are deterministic arithmetic, not runtime measurements. BGV packing depends on the chosen encoding and cryptographic parameters; no packing factor, latency or speedup is asserted.');
const footer6=records.find(v=>v.kind==='textbox'&&v.slide===6&&v.text==='6');if(footer6)p.resolve(footer6.id).text='7';
txt(old6,'Proposed quorum: 3 of 5 authorities. Two can be offline.',64,180,1120,36,25,C.teal,true);
const draft=build+'/candidate.pptx',final=root+'/outputs/threshold_encryption_bnp_dp_with_graphs.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);console.log('Draft saved');
await finalizePresentation({workspaceDir:root,candidatePath:draft,finalPath:final,pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','7'],requiredNativeTableOwnerSlides:[7],requiredNativeChartOwnerSlides:[4,5,6],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:build+'/validation.json'});
console.log('Finalized');const result=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<result.slides.items.length;i++){const png=await result.export({slide:result.slides.items[i],format:'png',scale:1});await fs.writeFile(build+`/slide-${i+1}.png`,new Uint8Array(await png.arrayBuffer()));console.log('Rendered',i+1);}



