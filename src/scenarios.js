import {nativeSystemPrompt,parseNativeAction} from './native-tools.js';
export const scenarios = [
  {id:'math-1', family:'Tool selection', title:'Order total', prompt:'Use the calculator to find the total cost of 7 items at 13 credits each plus 9 credits shipping. Finish with the numeric answer only.', expected:100},
  {id:'math-2', family:'Tool selection', title:'Remaining budget', prompt:'Use the calculator to find the remaining credits from a budget of 250 after buying 6 items at 27 credits each. Finish with the numeric answer only.', expected:88},
  {id:'craft-1', family:'Multi-step planning', title:'Build a shelter', prompt:'Build exactly one shelter. Gather what you need. Inspect inventory if useful. Finish when the shelter exists.', target:'shelter'},
  {id:'craft-2', family:'Multi-step planning', title:'Craft two torches', prompt:'Craft exactly two torches. Gather what you need. Inspect inventory if useful. Finish when both torches exist.', target:'torch'},
  {id:'recover-1', family:'Error recovery', title:'Blocked route', prompt:'Deliver parcel A to the depot. First attempt deliver with route="bridge"; if it fails, use the returned information to recover. Finish only after delivery.', parcel:'A'},
  {id:'recover-2', family:'Error recovery', title:'Blocked route, variant', prompt:'Deliver parcel B to the depot. First attempt deliver with route="tunnel"; if it fails, use the returned information to recover. Finish only after delivery.', parcel:'B'},
];

export const systemPrompt = `You control a tiny simulated world using tools. Return exactly ONE JSON object per turn, no markdown or reasoning.
To use a tool: {"tool":"tool_name","args":{...}}
To finish: {"final":"your answer"}
Wait for the tool result before choosing the next action. Do not invent tool results.
Available tools:
calculate: {"op":"add|subtract|multiply","a":number,"b":number}. One operation per call.
inventory: {}. Shows current resources and crafted items.
gather: {"resource":"wood|stone","amount":integer from 1 to 6}.
craft: {"item":"shelter|torch"}. One shelter costs 4 wood and 2 stone. One torch costs 2 wood.
deliver: {"parcel":"A|B","route":"bridge|tunnel|ferry"}. Delivers a parcel; errors contain recovery instructions.
Only perform actions needed for the user task. Tools operate in a sandbox, not the real world.`;

export function parseAction(text) {
  const cleaned = (text.includes('</think>')?answerOnly(text):text).trim().replace(/^```(?:json)?\s*/,'').replace(/\s*```$/,'');
  const value = JSON.parse(cleaned);
  if (!value || Array.isArray(value) || typeof value !== 'object') throw new Error('Expected a JSON object.');
  if (typeof value.final === 'string' && !('tool' in value) && Object.keys(value).length === 1) return value;
  if (typeof value.tool !== 'string' || !value.args || Array.isArray(value.args) || typeof value.args !== 'object' || 'final' in value || Object.keys(value).some(k=>!['tool','args'].includes(k))) throw new Error('Use {"tool":"name","args":{}} or {"final":"answer"}.');
  return value;
}

export function createWorld(scenario) {
  const state = {wood:0,stone:0,shelter:0,torch:0,delivered:[],deliveryAttempts:[],calculations:[]};
  function execute({tool,args}) {
    const keys = allowed => Object.keys(args).every(k=>allowed.includes(k));
    switch(tool) {
      case 'calculate': {
        const {op,a,b}=args;
        if(!keys(['op','a','b']) || !Number.isFinite(a) || !Number.isFinite(b) || !['add','subtract','multiply'].includes(op)) return {error:'Expected op add/subtract/multiply and finite numbers a,b.'};
        const value=op==='add'?a+b:op==='subtract'?a-b:a*b;
        if(!Number.isFinite(value)) return {error:'Numeric overflow.'};
        state.calculations.push(value); return {value};
      }
      case 'inventory': return keys([])?{...state}: {error:'inventory takes no arguments.'};
      case 'gather': {
        const {resource,amount}=args;
        if(!keys(['resource','amount']) || !['wood','stone'].includes(resource) || !Number.isInteger(amount) || amount<1 || amount>6) return {error:'Choose wood/stone and an integer amount from 1 to 6.'};
        state[resource]+=amount; return {resource,available:state[resource]};
      }
      case 'craft': {
        if(!keys(['item']) || !['shelter','torch'].includes(args.item)) return {error:'Unknown item.'};
        const wood=args.item==='shelter'?4:2, stone=args.item==='shelter'?2:0;
        if(state.wood<wood || state.stone<stone) return {error:'Insufficient resources.',need:{wood,stone},have:{wood:state.wood,stone:state.stone}};
        state.wood-=wood; state.stone-=stone; state[args.item]++; return {crafted:args.item,count:state[args.item]};
      }
      case 'deliver': {
        if(!keys(['parcel','route']) || args.parcel!==scenario.parcel || !['bridge','tunnel','ferry'].includes(args.route)) return {error:'Invalid parcel or route.'};
        state.deliveryAttempts.push(args.route);
        if(args.route!=='ferry') return {error:`The ${args.route} is closed. Retry deliver with route="ferry" and the same parcel.`};
        if(!state.delivered.includes(args.parcel)) state.delivered.push(args.parcel);
        return {delivered:args.parcel,destination:'depot'};
      }
      default: return {error:`Unknown tool: ${tool}`};
    }
  }
  function grade(final) {
    if(typeof final!=='string') return false;
    if(scenario.expected!==undefined) return final.trim()!=='' && Number(final.trim())===scenario.expected && state.calculations.includes(scenario.expected);
    if(scenario.target==='shelter') return state.shelter===1 && state.torch===0;
    if(scenario.target==='torch') return state.torch===2 && state.shelter===0;
    return state.delivered.includes(scenario.parcel) && state.deliveryAttempts[0]===(scenario.parcel==='A'?'bridge':'tunnel') && state.deliveryAttempts.at(-1)==='ferry';
  }
  return {state,execute,grade};
}

export function promptForMode(thinking=false){return thinking?systemPrompt.replace('Return exactly ONE JSON object per turn, no markdown or reasoning.','You may reason in the thinking block. After </think>, return exactly ONE JSON object per turn, no markdown.'):systemPrompt;}
export function answerOnly(text){return text.includes('</think>')?text.slice(text.lastIndexOf('</think>')+8).trim():text.startsWith('<think>')?'':text;}
export async function runScenario(scenario, generate, onStep=()=>{}, maxSteps=10, options={}) {
  const world=createWorld(scenario), messages=[{role:'system',content:options.protocol==='native'?nativeSystemPrompt:promptForMode(options.thinking)},{role:'user',content:scenario.prompt}], trace=[];
  const started=performance.now(); let final, terminationReason;
  for(let turn=1;turn<=maxSteps;turn++) {
    const generation=await generate(messages), step={turn,...generation};
    messages.push({role:'assistant',content:options.thinking?answerOnly(generation.text):generation.text});
    try {
      step.action=(options.protocol==='native'?parseNativeAction:parseAction)(generation.text);
      if('final' in step.action) final=step.action.final;
      else step.observation=structuredClone(world.execute(step.action));
    } catch(error) { step.observation={error:`Invalid action: ${error.message} ${options.protocol==='native'?'Return one complete native tool call.':'Return one JSON object.'}`}; step.parseError=true; }
    trace.push(step); onStep(step);
    if(final!==undefined) break;
    if(options.thinking && generation.finishReason==='length'){terminationReason='Thinking/output token limit reached';break;}
    messages.push(options.protocol==='native' && !step.parseError?{role:'tool',content:JSON.stringify(step.observation)}:{role:'user',content:`Tool result: ${JSON.stringify(step.observation)}`});
  }
  return {id:scenario.id,title:scenario.title,family:scenario.family,prompt:scenario.prompt,passed:world.grade(final),final:final??null,reason:terminationReason??(final===undefined?'Step limit reached':world.grade(final)?'Objective verified':'Objective not met'),durationMs:performance.now()-started,state:world.state,trace};
}
