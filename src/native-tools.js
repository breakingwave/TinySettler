import {actionSchema} from './action-schema.js';
const descriptions={
  calculate:'Perform one arithmetic operation. Use this tool for requested calculations.',
  inventory:'Inspect current wood, stone, shelter and torch counts.',
  gather:'Gather between 1 and 6 units of wood or stone.',
  craft:'Craft ONE item. One shelter consumes 4 wood and 2 stone. One torch consumes 2 wood.',
  deliver:'Deliver a parcel to the depot by the chosen route. A failed route returns instructions for recovery.',
};
export const nativeTools=actionSchema.oneOf.filter(x=>x.properties.tool).map(x=>({type:'function',function:{name:x.properties.tool.enum[0],description:descriptions[x.properties.tool.enum[0]],parameters:x.properties.args}}));
export const nativeSystemPrompt='You control a tiny simulated world. Use the provided tools to complete the user task. Make exactly one tool call per turn, then wait for its result. Never invent results or claim success before the tools confirm the objective. When finished, give a brief final answer.';
export function parseNativeAction(text){
  const clean=(text.includes('</think>')?text.slice(text.lastIndexOf('</think>')+8):text).trim();
  if(clean.startsWith('<think>'))throw new Error('Reasoning not finished.');
  const blocks=[...clean.matchAll(/<tool_call>([\s\S]*?)<\/tool_call>/g)];
  if(!blocks.length){if(!clean || /<\/?(?:tool_call|function|parameter)/.test(clean))throw new Error('Incomplete tool call.');return {final:clean};}
  if(blocks.length!==1 || clean.slice(blocks[0].index+blocks[0][0].length).trim())throw new Error('Return exactly one tool call, with no text after it.');
  const fn=blocks[0][1].trim().match(/^<function=([a-z_]+)>([\s\S]*)<\/function>$/);
  if(!fn)throw new Error('Invalid function block.');
  const tool=nativeTools.find(t=>t.function.name===fn[1]);if(!tool)throw new Error('Unknown tool.');
  const args={},body=fn[2];
  for(const m of body.matchAll(/<parameter=([a-z_]+)>([\s\S]*?)<\/parameter>/g)){
    const spec=tool.function.parameters.properties[m[1]];
    if(!spec || Object.hasOwn(args,m[1]))throw new Error('Unknown or duplicate parameter.');
    const value=m[2].trim();args[m[1]]=spec.type==='number'||spec.type==='integer'?Number(value):value;
    if(value==='' || ((spec.type==='number'||spec.type==='integer')&&!Number.isFinite(args[m[1]])))throw new Error('Invalid parameter value.');
  }
  if(body.replace(/<parameter=[a-z_]+>[\s\S]*?<\/parameter>/g,'').trim())throw new Error('Unexpected content in function.');
  if(tool.function.parameters.required.some(key=>!Object.hasOwn(args,key)))throw new Error('Missing required parameter.');
  return {tool:fn[1],args};
}
