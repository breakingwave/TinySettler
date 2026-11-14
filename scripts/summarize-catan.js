#!/usr/bin/env node
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';

const finite=values=>values.filter(Number.isFinite);
const median=values=>{const a=finite(values).sort((x,y)=>x-y);return a.length?(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2:null;};
const p95=values=>{const a=finite(values).sort((x,y)=>x-y);return a.length?a[Math.ceil(.95*a.length)-1]:null;};
const sum=values=>finite(values).reduce((a,b)=>a+b,0);
const countBy=(items,key)=>items.reduce((out,item)=>{const k=key(item);if(k)out[k]=(out[k]??0)+1;return out;},{});
const phase=decision=>decision.prompt?.match(/phase ([A-Z_]+)/)?.[1]??null;
const isSetup=decision=>phase(decision)?.startsWith('BUILD_INITIAL_')??false;
function trade(decision){
  const match=decision.action?.label?.match(/^trade (\d+) ([a-z]+) for (\d+) ([a-z]+)$/i);
  return match?{give:Number(match[1]),from:match[2].toLowerCase(),receive:Number(match[3]),to:match[4].toLowerCase()}:null;
}
function inverseTrades(decisions){
  const found=[];
  for(let i=1;i<decisions.length;i++){
    const a=trade(decisions[i-1]),b=trade(decisions[i]);
    if(!a||!b||decisions[i-1].turn!==decisions[i].turn||a.from!==b.to||a.to!==b.from)continue;
    found.push({decisionNumbers:[i,i+1],turn:decisions[i].turn,actions:[decisions[i-1].action.label,decisions[i].action.label],netResourceChange:{[a.from]:b.receive-a.give,[a.to]:a.receive-b.give},totalCardsLost:a.give+b.give-a.receive-b.receive});
  }
  return found;
}
export function summarizeCatanReport(report){
  const decisions=report.decisions??[],state=report.finalState??report.initialState,red=state?.players?.find(p=>p.color==='RED');
  const choices=decisions.filter(d=>(d.legalActions?.length??0)>1),first=choices.filter(d=>Number.isInteger(d.actionId)&&d.actionId===d.legalActions[0].id).length;
  const pairs=decisions.filter(d=>Number.isFinite(d.firstTokenMs)&&Number.isFinite(d.inferenceMs)&&d.inferenceMs>0),pairedElapsed=sum(pairs.map(d=>d.inferenceMs));
  const regular=decisions.filter(d=>!isSetup(d)),regularCounts=countBy(regular,d=>d.action?.type),allCounts=countBy(decisions,d=>d.action?.type);
  const timings=decisions.map(d=>d.inferenceMs),promptTokens=decisions.map(d=>d.usage?.prompt_tokens),outputTokens=decisions.map(d=>d.usage?.completion_tokens);
  const truncated=Boolean(!state?.winner&&(state?.done||/decision cap reached/i.test(report.error??''))),status=state?.winner?'winner':truncated?'truncated':report.error?'interrupted':report.completed?'completed':'in_progress';
  return {
    id:report.id,model:report.model,seed:report.seed,runtime:report.runtime,settings:report.settings,engine:state?.engine,status,completed:Boolean(report.completed),winner:state?.winner??null,truncated,reason:state?.reason??report.error??null,
    turn:state?.turn??null,redVP:red?.points??null,redBuildings:{settlements:red?.settlements??null,cities:red?.cities??null,roads:red?.roads??null,longestRoad:red?.longestRoad??null},
    decisions:decisions.length,invalid:decisions.filter(d=>d.error).length,wallSeconds:Number.isFinite(report.durationMs)?report.durationMs/1000:null,loadSeconds:Number.isFinite(report.loadMs)?report.loadMs/1000:null,
    latency:{medianDecisionMs:median(timings),p95DecisionMs:p95(timings),totalInferenceSeconds:sum(timings)/1000,medianFirstChunkMs:median(decisions.map(d=>d.firstTokenMs)),firstChunkFractionOfInferenceElapsed:pairedElapsed?sum(pairs.map(d=>d.firstTokenMs))/pairedElapsed:null,pairedObservations:pairs.length,medianDecodeTokensPerSecond:median(decisions.map(d=>d.usage?.extra?.decode_tokens_per_s)),medianPrefillTokensPerSecond:median(decisions.map(d=>d.usage?.extra?.prefill_tokens_per_s))},
    tokens:{promptTotal:sum(promptTokens),promptMedian:median(promptTokens),outputTotal:sum(outputTokens),outputMedian:median(outputTokens),reportedTotal:sum(decisions.map(d=>d.usage?.total_tokens))},
    modelActionCounts:allCounts,engineRedActionCounts:state?.stats?.redActionCounts??null,forcedActions:state?.stats?.forcedActions??null,
    nonSetupModelActions:{settlements:regularCounts.BUILD_SETTLEMENT??0,cities:regularCounts.BUILD_CITY??0,roads:regularCounts.BUILD_ROAD??0,developmentPurchases:regularCounts.BUY_DEVELOPMENT_CARD??0,developmentPlays:sum(Object.entries(regularCounts).filter(([k])=>k.startsWith('PLAY_')).map(([,n])=>n)),trades:regularCounts.MARITIME_TRADE??0,endTurns:regularCounts.END_TURN??0},
    firstListedChoice:{count:first,eligibleDecisions:choices.length,ratio:choices.length?first/choices.length:null,setupCount:choices.filter(d=>isSetup(d)&&d.actionId===d.legalActions[0].id).length,setupEligible:choices.filter(isSetup).length},
    consecutiveInverseTrades:inverseTrades(decisions),
  };
}
async function main(){
  const args=process.argv.slice(2),write=args.includes('--write');
  if(args.some(x=>x.startsWith('--')&&x!=='--write'))throw new Error('Usage: node scripts/summarize-catan.js [report.json ...] [--write]');
  const files=args.filter(x=>x!=='--write');if(!files.length)files.push('catan-results/latest.json');
  const byId=new Map();
  for(const file of files){
    const data=JSON.parse(await readFile(file,'utf8')),reports=Array.isArray(data.results)?data.results:Array.isArray(data.decisions)?[data]:[];
    if(!reports.length)throw new Error(`No Catan game reports found in ${file}`);
    for(const report of reports){if(!Array.isArray(report.decisions)||!report.model)throw new Error(`Invalid Catan report in ${file}`);const id=report.id??`${file}:${byId.size}`,old=byId.get(id);if(!old||(report.decisions.length>=old.report.decisions.length))byId.set(id,{report,source:file});}
  }
  const summary={schemaVersion:1,generatedAt:new Date().toISOString(),notes:[
    'Legal action IDs are constrained by the decoding grammar. A low invalid count does not establish strategy or unconstrained protocol compliance. The invalid field counts recorded decision errors, including parse or action-API failures.',
    'Non-setup action counts include model-selected actions only; engineRedActionCounts also includes automatically executed forced actions.',
    'First listed choice ratio excludes decisions with only one legal action. It is descriptive, not proof of positional bias.',
    'Consecutive inverse trades are a heuristic: adjacent model decisions in the same turn exchanging the same two resources in opposite directions. Not all trades are bad.',
    'First-chunk timing includes prompt prefill and runtime/grammar overhead; it is not an isolated measurement of GPU prefill.',
    'A seed matches the initial random setup; different actions can consume random draws differently. Small samples do not establish playing strength.'
  ],games:[...byId.values()].map(({report,source})=>({...summarizeCatanReport(report),source}))};
  if(write){const target=resolve('catan-results/summary.json');if(files.some(file=>resolve(file)===target))throw new Error('Refusing to overwrite an input file');await mkdir(dirname(target),{recursive:true});await writeFile(target,JSON.stringify(summary,null,2)+'\n');}
  process.stdout.write(JSON.stringify(summary,null,2)+'\n');
}
if(process.argv[1]&&resolve(process.argv[1])===resolve(fileURLToPath(import.meta.url)))main().catch(error=>{process.stderr.write(error.message+'\n');process.exitCode=1;});
