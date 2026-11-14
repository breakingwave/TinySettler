import {scenarios,createWorld} from '../src/scenarios.js';
import {readFile} from 'node:fs/promises';
const median = a => {a=a.filter(Number.isFinite).sort((x,y)=>x-y);return a.length?(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2:null;};
const reached=r=>{const scenario=scenarios.find(s=>s.id===r.id);if(!scenario)return null;const world=createWorld(scenario);Object.assign(world.state,r.state);return world.grade(String(scenario.expected??'Done'));};
for(const file of process.argv.slice(2)) {
  const r=JSON.parse(await readFile(file,'utf8'));if(!Array.isArray(r.results))continue;const steps=r.results.flatMap(x=>x.trace);
  console.log(JSON.stringify({file,model:r.model,mode:r.settings.outputMode,protocol:r.settings.protocol??'json',thinking:r.settings.thinking,completed:r.completed,passed:r.results.filter(x=>x.passed).length,total:r.results.length,toolObjectivesReached:r.results.filter(reached).length,turns:steps.length,parseErrors:steps.filter(x=>x.parseError).length,medianFirstTokenMs:median(steps.map(x=>x.firstTokenMs)),medianDecodeTokensPerSecond:median(steps.map(x=>x.usage?.extra?.decode_tokens_per_s)),durationSeconds:r.durationMs/1000,loadSeconds:r.loadMs/1000,results:r.results.map(x=>({id:x.id,passed:x.passed,toolObjectiveReached:reached(x),turns:x.trace.length,final:x.final,reason:x.reason}))},null,2));
}
