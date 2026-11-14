#!/usr/bin/env node
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {summarizeCatanReport} from './summarize-catan.js';

const finite=a=>a.filter(Number.isFinite);
const median=a=>{a=finite(a).sort((x,y)=>x-y);return a.length?(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2:null;};
const p95=a=>{a=finite(a).sort((x,y)=>x-y);return a.length?a[Math.ceil(a.length*.95)-1]:null;};
const sum=a=>finite(a).reduce((a,b)=>a+b,0);
const countBy=(a,key)=>a.reduce((o,v)=>{const k=key(v);if(k)o[k]=(o[k]??0)+1;return o;},{});
const budgetStops=new Set(['token_limit','decode_deadline','stage_deadline']);
const isThinking=r=>r.settings?.thinking===true||(r.decisions??[]).some(d=>d.thinking?.enabled===true);
const measured=(values)=>({observations:finite(values).length,median:median(values),p95:p95(values),total:sum(values)});

function lastObservedBefore(report,turnLimit=160){
  const candidates=(report.decisions??[]).map((d,i)=>({decision:i+1,turn:d.turn,redVP:Number(d.prompt?.match(/(?:^|[;\s])RED=(\d+)VP/)?.[1]),source:'model input observation'})).filter(x=>Number.isFinite(x.turn)&&x.turn<=turnLimit&&Number.isFinite(x.redVP));
  const final=report.finalState;if(final?.turn<=turnLimit){const red=final.players?.find(p=>p.color==='RED');if(Number.isFinite(red?.points))candidates.push({decision:null,turn:final.turn,redVP:red.points,source:'final engine state'});}
  return candidates.at(-1)??null;
}

export function summarizeThinkingReport(report){
  const base=summarizeCatanReport(report),decisions=report.decisions??[],thinking=decisions.filter(d=>d.thinking?.enabled===true),selection=decisions.filter(d=>d.selection);
  const latencies=decisions.map(d=>d.inferenceMs),med=median(latencies);
  return {...base,
    observationVersion:report.initialState?.observationVersion??report.finalState?.observationVersion??'unversioned',
    target:{medianDecisionMs:30000,measuredMedianMs:med,met:med===null?null:med<=30000,decisionsOver30Seconds:decisions.filter(d=>Number.isFinite(d.inferenceMs)&&d.inferenceMs>30000).length,measuredDecisions:finite(latencies).length},
    reasoning:{measuredDecisions:thinking.length,methods:[...new Set(thinking.map(d=>d.thinking.method).filter(Boolean))],cachePaths:[...new Set(thinking.map(d=>d.thinking.cachePath).filter(Boolean))],samplingConfigurations:[...new Map(thinking.filter(d=>d.thinking.sampling).map(d=>[JSON.stringify(d.thinking.sampling),d.thinking.sampling])).values()],
      configuredTokenBudgets:countBy(thinking,d=>String(d.thinking.budgetTokens??'unrecorded')),
      configuredDecodeBudgetsMs:countBy(thinking,d=>String(d.thinking.decodeBudgetMs??'unrecorded')),
      configuredStageBudgetsMs:countBy(thinking,d=>String(d.thinking.stageBudgetMs??'unrecorded')),
      generatedTokens:measured(thinking.map(d=>d.thinking.actualTokens??d.thinking.usage?.completion_tokens)),
      durationMs:measured(thinking.map(d=>d.thinking.durationMs)),firstChunkMs:measured(thinking.map(d=>d.thinking.firstTokenMs)),
      naturalClosures:thinking.filter(d=>d.thinking.reasoningClosedNaturally===true).length,
      forcedBoundaries:thinking.filter(d=>d.thinking.reasoningClosedNaturally===false).length,
      budgetTerminations:thinking.filter(d=>budgetStops.has(d.thinking.cutoffReason)).length,
      terminationReasons:countBy(thinking,d=>d.thinking.cutoffReason??'unrecorded'),
      promptTokens:measured(thinking.map(d=>d.thinking.usage?.prompt_tokens)),
    },
    selection:{measuredDecisions:selection.length,durationMs:measured(selection.map(d=>d.selection.durationMs)),firstChunkMs:measured(selection.map(d=>d.selection.firstTokenMs)),promptTokens:measured(selection.map(d=>d.selection.usage?.prompt_tokens)),outputTokens:measured(selection.map(d=>d.selection.usage?.completion_tokens)),finishReasons:countBy(selection,d=>d.selection.finishReason??'unrecorded'),terminationReasons:countBy(selection,d=>d.selection.cutoffReason??'unrecorded'),configuredStageBudgetsMs:countBy(selection,d=>String(d.selection.stageBudgetMs??'unrecorded'))},
    lastObservedAtOrBeforeOriginalTurnCap:lastObservedBefore(report),
  };
}

export function summarizeThinkingReports(reports,{completedOnly=false}={}){
  const thinkingReports=reports.filter(isThinking),excludedIncomplete=completedOnly?thinkingReports.filter(r=>r.completed!==true):[],selected=completedOnly?thinkingReports.filter(r=>r.completed===true):thinkingReports,games=selected.map(summarizeThinkingReport),models=[...new Set(selected.map(r=>r.model))].map(model=>{
    const matching=selected.filter(r=>r.model===model),steps=matching.flatMap(r=>r.decisions??[]),times=steps.map(d=>d.inferenceMs),med=median(times);
    return {model,games:matching.length,seeds:matching.map(r=>r.seed),redWins:matching.filter(r=>r.finalState?.winner==='RED').length,redPoints:matching.map(r=>r.finalState?.players?.find(p=>p.color==='RED')?.points??null),decisions:steps.length,invalid:steps.filter(d=>d.error).length,medianDecisionMs:med,p95DecisionMs:p95(times),medianWithin30Seconds:med===null?null:med<=30000,reasoningTokenMedian:median(steps.map(d=>d.thinking?.actualTokens)),reasoningNaturalClosures:steps.filter(d=>d.thinking?.reasoningClosedNaturally===true).length,reasoningBudgetTerminations:steps.filter(d=>budgetStops.has(d.thinking?.cutoffReason)).length};
  });
  return {schemaVersion:2,generatedAt:new Date().toISOString(),selectionPolicy:completedOnly?'completed-thinking-reports-only':'all-thinking-reports',excludedNonThinkingReports:reports.length-thinkingReports.length,excludedIncompleteReports:excludedIncomplete.length,notes:[
    completedOnly?'Only reports explicitly marked completed enter games and model aggregates. Incomplete/calibration reports remain summarized separately in excludedIncompleteGames. All attempts within retained reports, including errors and slow decisions, remain included.':'No incomplete thinking report is automatically excluded. Use --completed-only or explicit final report filenames to separate calibration from full-run aggregates.',
    'End-to-end decision latency uses the outer inferenceMs measurement and includes both stages, prompt processing, resets and runtime overhead. Load time is excluded. Report the median and tail separately; a median under30s does not guarantee every action under30s.',
    'Thinking is genuinely generated in a bounded first pass; a separate constrained selection pass receives that draft. Forced boundaries and budget terminations are recorded. This is budgeted reasoning, not unrestricted chain-of-thought.',
    'All measured attempts, including recorded errors, remain in latency statistics. A game that stops before recording an attempt may have additional unmeasured waiting time, visible in wallSeconds.',
    'The constrained legal-action grammar makes legal output largely guaranteed; score, construction and wins assess strategy separately.',
    'The old no-thinking reports used a lower turn cap, observation v1 and a different Python hash-seed configuration. Those outcomes are historical references, not an isolated causal ablation of thinking.',
    'lastObservedAtOrBeforeOriginalTurnCap is the last saved model-input or final-state observation at or before turn160. It is not necessarily the exact state at turn160.',
    'Two-stage token usage counts both prompt evaluations. First-chunk latencies include prefill and runtime overhead; they are not isolated GPU timings.'
  ],models,games,excludedIncompleteGames:excludedIncomplete.map(summarizeThinkingReport)};
}

async function main(){
  const args=process.argv.slice(2),write=args.includes('--write'),completedOnly=args.includes('--completed-only');if(args.some(a=>a.startsWith('--')&&!['--write','--completed-only'].includes(a)))throw Error('Usage: node scripts/summarize-catan-thinking.js [reports.json ...] [--completed-only] [--write]');
  const files=args.filter(a=>!['--write','--completed-only'].includes(a));if(!files.length)files.push('catan-results/latest.json');const byId=new Map();
  for(const file of files){const data=JSON.parse(await readFile(file,'utf8'));const reports=Array.isArray(data.results)?data.results:Array.isArray(data.decisions)?[data]:[];if(!reports.length)throw Error(`No game reports in ${file}`);for(const report of reports){const key=report.id??`${file}:${byId.size}`,old=byId.get(key);if(!old||(report.decisions?.length??0)>=(old.report.decisions?.length??0))byId.set(key,{report,source:file});}}
  const entries=[...byId.values()],summary=summarizeThinkingReports(entries.map(e=>e.report),{completedOnly});for(const game of [...summary.games,...summary.excludedIncompleteGames])game.source=entries.find(e=>e.report.id===game.id)?.source;
  if(write){const target=resolve('catan-results/thinking-summary.json');if(files.some(file=>resolve(file)===target))throw Error('Refusing to overwrite an input report');await mkdir(dirname(target),{recursive:true});await writeFile(target,JSON.stringify(summary,null,2)+'\n');}
  process.stdout.write(JSON.stringify(summary,null,2)+'\n');
}
if(process.argv[1]&&resolve(process.argv[1])===resolve(fileURLToPath(import.meta.url)))main().catch(e=>{process.stderr.write(e.message+'\n');process.exitCode=1;});
