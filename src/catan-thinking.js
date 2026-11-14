export const THINKING_PREFIX='Compare legal actions: ';
export function boundedDraft(rawText){const closeIndex=rawText.indexOf('</think>');return THINKING_PREFIX+(closeIndex>=0?rawText.slice(0,closeIndex):rawText);}
/** Isolated current-decision reasoning; no draft carries into the next game action. */
export function boundedReasoningMessages(messages,budgetTokens=256){
  const copy=messages.map(message=>({...message}));
  const instruction=`\nThinking phase: use the opened thinking block to compare at most two legal actions and choose a concrete actionId. Be concise: roughly ${Math.max(8,Math.floor(budgetTokens*.55))} words, then close </think>. Do not restate the board or rules. Check immediate victory, useful building/resource trades, and avoid reversing a previous trade. A later constrained stage will emit the action JSON. This thinking-phase instruction overrides instructions to immediately output only JSON.`;
  const system=copy.find(message=>message.role==='system');
  if(system)system.content+=instruction;else copy.unshift({role:'system',content:instruction.trim()});
  return copy;
}
export function selectionMessages(messages,draft){
  const copy=messages.map(message=>({...message}));
  const last=copy.at(-1);if(!last||last.role!=='user')throw new Error('Expected a current user observation.');
  // The same model receives its own draft as task data. No additional policy picks an action.
  last.content+=`\n\nYour reasoning draft for this decision (it may be cut off):\n<bounded_draft>\n${draft}\n</bounded_draft>\nThe thinking budget is finished. Use the observation and draft to choose ONE currently listed legal action. Return only {"actionId":number}. Do not continue reasoning.`;
  return copy;
}
export function combineStageUsage(...stages){
  const present=stages.filter(Boolean);if(!present.length)return undefined;
  return {prompt_tokens:present.reduce((sum,s)=>sum+(s.prompt_tokens??0),0),completion_tokens:present.reduce((sum,s)=>sum+(s.completion_tokens??0),0),total_tokens:present.reduce((sum,s)=>sum+(s.total_tokens??((s.prompt_tokens??0)+(s.completion_tokens??0))),0),extra:{stages:present.length,decode_tokens_per_s:null,prefill_tokens_per_s:null}};
}
