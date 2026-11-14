import {nativeSystemPrompt,nativeTools} from './native-tools.js';
import './style.css';
import {scenarios,runScenario,promptForMode} from './scenarios.js';

document.querySelector('#app').innerHTML=`
<header><a class="brand" href="/">◈ <span>TinySettler</span></a><span class="tag">BROWSER AGENT LAB</span><a class="local" href="/catan.html">Catan agents ↗</a></header>
<main><section class="intro"><div class="eyebrow">EXPERIMENT 01 / SMALL LANGUAGE MODELS</div><h1>Small model.<br><span>Real actions.</span></h1><p>Can a two-billion-parameter model use tools, make a plan, and recover from an error? Run it on your own GPU.</p></section>
<section class="setup panel"><div class="section-heading"><h2>01 <span>Runtime</span></h2><span id="badge" class="tag">CHECKING GPU</span></div><div class="settings"><label>Model<select id="model"><option value="Qwen3.5-2B-q4f16_1-MLC">Qwen3.5 · 2B · 4-bit</option><option value="Qwen3.5-2B-q4f32_1-MLC">Qwen3.5 · 2B · 4-bit / FP32</option><option value="onnx:q4f16">Qwen3.5 · ONNX · 4-bit / FP16</option><option value="onnx:fp16">Qwen3.5 · ONNX · FP16 weights</option><option value="Qwen3-1.7B-q4f16_1-MLC">Qwen3 · 1.7B · 4-bit</option><option value="Llama-3.2-1B-Instruct-q4f16_1-MLC">Llama 3.2 · 1B · 4-bit</option></select></label><button id="load">Load model <span>↗</span></button></div><p id="gpu" class="muted">Detecting WebGPU…</p><p class="muted">WebGPU → Metal on macOS. Model weights download once and are cached by this browser. Allow several GB of free memory.</p><progress id="progress" max="1" value="0"></progress><p id="status" role="status">Ready to load. No API key or inference server.</p></section>
<section class="suite"><div class="section-heading"><h2>02 <span>Agent scenarios</span></h2><label class="repeat">Protocol <select id="protocol"><option value="json">Custom JSON</option><option value="native">Qwen native tools (ONNX)</option></select></label><label class="repeat">Thinking <select id="thinking"><option value="off">Off</option><option value="on">On · 2048 tokens</option></select></label><label class="repeat">Sampling <select id="sampling"><option value="greedy">Greedy</option><option value="qwen">Sampled · T1 / p0.95</option></select></label><label class="repeat">Output <select id="output-mode"><option value="schema">JSON schema</option><option value="prompt-only">Prompt only</option></select></label><label class="repeat">Runs per scenario <select id="repeats"><option>1</option><option>3</option><option>5</option></select></label></div><div class="cards">${['Tool selection','Multi-step planning','Error recovery'].map((family,i)=>`<article class="panel"><span class="card-num">0${i+1}</span><h3>${family}</h3><p>${['Choose calculator operations and return the observed result.','Gather resources and craft items in a tiny settlement.','Observe a blocked route and retry using the tool feedback.'][i]}</p><span class="tag">2 TASKS · VERIFIED OBJECTIVES</span></article>`).join('')}</div><div class="actions"><button id="run" disabled>Run all 6 scenarios <span>→</span></button><button id="stop" class="secondary" disabled>Stop & unload</button><button id="export" class="secondary" disabled>Export JSON ↓</button><span class="muted">10 turns max · thinking uses prompt-only output</span></div></section>
<section class="metrics"><div><span class="label">SUCCESS</span><strong id="score">—</strong></div><div><span class="label">MEDIAN FIRST TOKEN</span><strong id="ttft">—</strong></div><div><span class="label">DECODE SPEED</span><strong id="speed">—</strong></div><div><span class="label">TOTAL RUN TIME</span><strong id="elapsed">—</strong></div></section>
<section class="panel results"><div class="section-heading"><h2>03 <span>Results & traces</span></h2><span id="count" class="tag">NO RUNS YET</span></div><p class="muted" id="empty">Results will appear here. Expand a task to inspect every model action and tool response.</p><div id="results"></div><div id="livebox" hidden><span class="label" id="live-label">LIVE MODEL OUTPUT</span><pre id="live"></pre></div></section>
<footer>WebLLM 0.2.87 / Transformers.js 4.3.2 · Simulated tools only · No prompts sent to a model server.<br>Small smoke test, not a general agent benchmark. Repeated greedy runs measure stability, not independent samples.</footer></main>`;
const $=id=>document.getElementById(id);
let worker, seq=0, pending=new Map(), ready=false, busy=false, report, adapterInfo, loadMs, runStart, activeModel;
function status(text){$('status').textContent=text;}
function controls(){if(!$('model').value.startsWith('onnx:'))$('protocol').value='json';$('protocol').disabled=busy||!$('model').value.startsWith('onnx:'); $('load').disabled=busy||ready; $('model').disabled=busy||ready; $('run').disabled=busy||!ready; $('repeats').disabled=busy; $('thinking').disabled=busy; if($('model').value.startsWith('onnx:'))$('sampling').value='greedy'; $('sampling').disabled=busy||$('model').value.startsWith('onnx:'); const promptOnly=$('thinking').value==='on'||$('model').value.startsWith('onnx:');if(promptOnly)$('output-mode').value='prompt-only';$('output-mode').disabled=busy||promptOnly; $('stop').disabled=!worker; $('export').disabled=!report; }
function rpc(type,extra={},timeout=180000){return new Promise((resolve,reject)=>{const id=++seq;const timer=setTimeout(()=>{unload('Timed out. Model unloaded; reload to retry.');},timeout);pending.set(id,{resolve,reject,timer});worker.postMessage({id,type,...extra});});}
function unload(reason){worker?.terminate();worker=null;ready=false;busy=false;for(const p of pending.values()){clearTimeout(p.timer);p.reject(new Error(reason));}pending.clear();$('badge').textContent='UNLOADED';status(reason);controls();}
async function detect(){
  if(!navigator.gpu)throw new Error('WebGPU is unavailable. Open this localhost app in a current GPU-enabled browser.');
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  if(!adapter)throw new Error('No WebGPU adapter. Check browser hardware acceleration.');
  if(!adapter.features.has('shader-f16'))throw new Error('This model build requires shader-f16 support.');
  const info=adapter.info;
  adapterInfo={vendor:info?.vendor,architecture:info?.architecture,device:info?.device,description:info?.description,isFallbackAdapter:info?.isFallbackAdapter,features:[...adapter.features],maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,userAgent:navigator.userAgent};
  $('gpu').textContent=`GPU: ${[info?.vendor,info?.architecture,info?.description].filter(Boolean).join(' / ')||'adapter available'} · shader-f16 available`;
  $('badge').textContent='WEBGPU AVAILABLE';
}
$('load').onclick=async()=>{
  busy=true;activeModel=$('model').value;controls();status('Initializing runtime…');const start=performance.now();
  worker=activeModel.startsWith('onnx:')?new Worker(new URL('./transformers.worker.js',import.meta.url),{type:'module'}):new Worker(new URL('./inference.worker.js',import.meta.url),{type:'module'}); controls();
  worker.onmessage=({data})=>{
    if(data.type==='progress'){$('progress').value=data.progress.progress;status(data.progress.text);return;}
    if(data.type==='token'){$('live').textContent=data.text;return;}
    const p=pending.get(data.id);if(!p)return;clearTimeout(p.timer);pending.delete(data.id);data.error?p.reject(new Error(data.error)):p.resolve(data.result);
  };
  worker.onerror=event=>unload(`Worker error: ${event.message}`);
  try{await rpc('load',{model:activeModel},900000);loadMs=performance.now()-start;ready=true;busy=false;$('badge').textContent='MODEL READY';status(`Loaded in ${(loadMs/1000).toFixed(1)} s. Ready to run.`);controls();}
  catch(error){unload(error.message);}
};
$('thinking').onchange=controls; $('model').onchange=controls;
$('stop').onclick=()=>{$('livebox').hidden=true;unload('Stopped. Model unloaded. Cached downloads are retained.');};
function median(values){const a=values.filter(Number.isFinite).sort((a,b)=>a-b);return a.length?(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2:null;}
function updateMetrics(){
  const results=report.results,steps=results.flatMap(r=>r.trace),t=median(steps.map(s=>s.firstTokenMs)),speed=median(steps.map(s=>s.usage?.extra?.decode_tokens_per_s));
  $('score').textContent=`${results.filter(r=>r.passed).length} / ${results.length}`;
  $('ttft').textContent=t===null?'—':`${(t/1000).toFixed(2)} s`;
  $('speed').textContent=speed===null?'—':`${speed.toFixed(1)} tok/s`;
  $('elapsed').textContent=`${((report.durationMs??(performance.now()-runStart))/1000).toFixed(1)} s`;
  $('count').textContent=`${results.length} / ${report.plannedRuns} COMPLETED`; $('empty').hidden=false; $('empty').textContent=`${report.model} · ${report.settings.protocol??'json'} · ${report.settings.outputMode??'prompt-only'} · ${report.completed?'Completed':report.error?'Interrupted':'In progress'}`;
}
function addResult(result){
  const details=document.createElement('details'),summary=document.createElement('summary');
  summary.textContent=`${result.passed?'PASS':'FAIL'}   ${result.title} · run ${result.repeat}   /   ${result.trace.length} turns   /   ${(result.durationMs/1000).toFixed(1)} s`;
  summary.className=result.passed?'pass':'fail';details.append(summary);
  const pre=document.createElement('pre');pre.textContent=JSON.stringify({reason:result.reason,final:result.final,state:result.state,trace:result.trace},null,2);details.append(pre);$('results').append(details);
}
async function save(){
  try{localStorage.setItem('tinysettler-latest',JSON.stringify(report));}catch{report.storageWarning='Browser storage unavailable; export JSON to retain results.';}
  if(import.meta.env.DEV){try{const response=await fetch('/__results',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)});report.savedToProject=response.ok;}catch{report.savedToProject=false;}}
}
$('run').onclick=async()=>{
  busy=true;controls();runStart=performance.now();$('results').replaceChildren();$('empty').hidden=true;$('livebox').hidden=false;
  report={app:'TinySettler',schemaVersion:1,startedAt:new Date().toISOString(),model:activeModel,runtime:activeModel.startsWith('onnx:')?'Transformers.js 4.3.2 / ONNX Runtime Web':'WebLLM 0.2.87',modelRepository:activeModel.startsWith('onnx:')?'onnx-community/Qwen3.5-2B-ONNX-OPT':'mlc-ai/'+activeModel,backend:'WebGPU',metal:adapterInfo.architecture?.includes('metal')?'Browser adapter reports Metal architecture':'Not confirmed by adapter metadata',adapter:adapterInfo,systemPrompt:$('protocol').value==='native'?nativeSystemPrompt:promptForMode($('thinking').value==='on'),toolDefinitions:$('protocol').value==='native'?nativeTools:undefined,settings:{protocol:$('protocol').value,outputMode:$('output-mode').value,temperature:$('sampling').value==='qwen'?1:0,topP:$('sampling').value==='qwen'?0.95:1,presencePenalty:$('sampling').value==='qwen'?1.5:0,topK:null,seed:42,thinking:$('thinking').value==='on',maxTokens:$('thinking').value==='on'?2048:160,contextWindow:activeModel.startsWith('onnx:')?null:8192,assistantPrefix:$('thinking').value==='on'?'<think>\n':'<think>\n\n</think>\n\n',chatTemplate:$('protocol').value==='native'?'Tokenizer native Qwen3.5 tool template':'MLC-compatible text-only ChatML',history:$('thinking').value==='on'?'answers-only/full-prefill':activeModel.startsWith('onnx:')?'full-prefill':'cached',maxSteps:10,repeats:Number($('repeats').value)},loadMs,plannedRuns:6*Number($('repeats').value),results:[],completed:false};
  updateMetrics();
  try{
    for(let repeat=1;repeat<=report.settings.repeats;repeat++)for(const scenario of scenarios){
      status(`Running ${scenario.title} · repetition ${repeat}`);$('live-label').textContent=`${scenario.title.toUpperCase()} · LIVE MODEL OUTPUT`;
      await rpc('reset');
      const result=await runScenario(scenario,messages=>{ $('live').textContent='Generating…';return rpc('generate',{messages,protocol:report.settings.protocol,outputMode:report.settings.outputMode,thinking:report.settings.thinking,maxTokens:report.settings.maxTokens,temperature:report.settings.temperature,topP:report.settings.topP,presencePenalty:report.settings.presencePenalty}); },step=>{$('live-label').textContent=`${scenario.title.toUpperCase()} · TURN ${step.turn} · ${step.observation?.error?'TOOL ERROR':step.action?.tool??'FINAL'}`;},10,{thinking:report.settings.thinking,protocol:report.settings.protocol});
      result.repeat=repeat;report.results.push(result);addResult(result);updateMetrics();await save();
    }
    report.completed=true;status('Suite complete. Expand a result to inspect its trace.');
  }catch(error){report.error=error.message;status(`Run ended: ${error.message}`);}
  finally{report.durationMs=performance.now()-runStart;report.finishedAt=new Date().toISOString();try{await save();}catch(error){status(`Results available; saving failed: ${error.message}`);}busy=false;updateMetrics();controls();}
};
$('export').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`tinysettler-${Date.now()}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
detect().catch(error=>{status(error.message);$('badge').textContent='GPU UNAVAILABLE';$('load').disabled=true;});

try{const saved=JSON.parse(localStorage.getItem('tinysettler-latest'));if(saved?.app==='TinySettler' && Array.isArray(saved.results)){report=saved;if(!report.completed){report.error??='Previous run interrupted';report.durationMs??=report.results.reduce((sum,r)=>sum+r.durationMs,0);}report.results.forEach(addResult);updateMetrics();controls();}}catch{/* Storage may be unavailable. */}
