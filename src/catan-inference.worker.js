import {CreateMLCEngine,prebuiltAppConfig} from '@mlc-ai/web-llm';
import {boundedReasoningMessages,selectionMessages,combineStageUsage,THINKING_PREFIX,boundedDraft} from './catan-thinking.js';
let engine,loadedModel;

// Cooperative deadlines: WebLLM observes interruption between GPU/token steps.
// A deadline cannot cancel an already-running prefill kernel.
async function collect(request,{phase,stageMs,decodeMs,onText,stopAtThinkClose=false}){
  const started=performance.now();let text='',firstTokenMs=null,usage,finishReason,cutoffReason=null,naturalClose=false,decodeTimer;
  const interrupt=reason=>{if(cutoffReason===null){cutoffReason=reason;void engine.interruptGenerate();}};
  const stageTimer=setTimeout(()=>interrupt('stage_deadline'),stageMs);
  try{
    const stream=await engine.chat.completions.create({...request,stream:true,stream_options:{include_usage:true}});
    for await(const chunk of stream){
      const part=chunk.choices?.[0]?.delta?.content;
      if(part){
        if(firstTokenMs===null){firstTokenMs=performance.now()-started;if(decodeMs)decodeTimer=setTimeout(()=>interrupt('decode_deadline'),decodeMs);}
        text+=part;
        if(stopAtThinkClose && text.includes('</think>')){naturalClose=true;interrupt('natural_close');}
        onText?.(text);
      }
      if(chunk.usage)usage=chunk.usage;
      if(chunk.choices?.[0]?.finish_reason)finishReason=chunk.choices[0].finish_reason;
    }
  }finally{clearTimeout(stageTimer);clearTimeout(decodeTimer);}
  if(cutoffReason===null)cutoffReason=finishReason==='length'?'token_limit':'model_stop';
  return {phase,text,firstTokenMs,durationMs:performance.now()-started,usage,finishReason,cutoffReason,naturalClose};
}
self.onmessage=async({data})=>{
  const {id,type}=data;
  try{
    if(type==='load'){
      loadedModel=data.model;
      const record=prebuiltAppConfig.model_list.find(m=>m.model_id===data.model);
      engine=await CreateMLCEngine(data.model,{initProgressCallback:progress=>self.postMessage({type:'progress',progress})},{context_window_size:8192,conv_config:{...record?.overrides?.conv_config,role_empty_sep:'\n<think>\n'+THINKING_PREFIX}});
      self.postMessage({id,result:{vendor:await engine.getGPUVendor()}});
    }else if(type==='reset'){
      await engine.resetChat();self.postMessage({id,result:true});
    }else if(type==='generate'){
      const started=performance.now();
      if(!data.thinking){
        const selection=await collect({messages:data.messages,temperature:0,top_p:1,max_tokens:data.maxTokens??32,seed:42,extra_body:{enable_thinking:false},response_format:{type:'json_object',schema:JSON.stringify(data.schema)}},{phase:'selection',stageMs:data.selectionStageMs??30000,onText:text=>self.postMessage({type:'token',phase:'selection',text})});
        self.postMessage({id,result:{...selection,durationMs:performance.now()-started}});return;
      }
      const budgetTokens=Math.max(16,Math.min(512,data.reasoningTokens??256));
      const decodeBudgetMs=Math.max(1000,Math.min(15000,data.reasoningDecodeMs??12000));
      const stageBudgetMs=Math.max(1000,Math.min(20000,data.reasoningWallMs??data.reasoningStageMs??18000));
      const temperature=data.reasoningTemperature??(loadedModel?.includes('Qwen3.5')?1:0.6);
      const topP=data.reasoningTopP??0.95;
      const presencePenalty=data.reasoningPresencePenalty??(loadedModel?.includes('Qwen3.5')?1.5:0);
      await engine.resetChat();
      const reasoningMessages=boundedReasoningMessages(data.messages,budgetTokens);
      const reasoning=await collect({messages:reasoningMessages,temperature,top_p:topP,presence_penalty:presencePenalty,max_tokens:budgetTokens,seed:42,extra_body:{enable_thinking:true}},{phase:'thinking',stageMs:stageBudgetMs,decodeMs:decodeBudgetMs,stopAtThinkClose:true,onText:text=>self.postMessage({type:'token',phase:'thinking',text:'<think>\n'+THINKING_PREFIX+text})});
      const draft=boundedDraft(reasoning.text);
      // Public chat completions cannot safely continue a length/abort finish in-place.
      // Rebuild a second prompt that explicitly receives the actual bounded draft.
      await engine.resetChat();
      const finalMessages=selectionMessages(data.messages,draft);
      const selection=await collect({messages:finalMessages,temperature:0,top_p:1,max_tokens:data.maxTokens??32,seed:42,extra_body:{enable_thinking:false},response_format:{type:'json_object',schema:JSON.stringify(data.schema)}},{phase:'selection',stageMs:data.selectionStageMs??30000,onText:text=>self.postMessage({type:'token',phase:'selection',text:'<think>\n'+draft+'\n</think>\n\n'+text})});
      self.postMessage({id,result:{text:selection.text,firstTokenMs:reasoning.firstTokenMs,durationMs:performance.now()-started,finishReason:selection.finishReason,usage:combineStageUsage(reasoning.usage,selection.usage),thinking:{enabled:true,method:'bounded-draft-then-constrained-selection',cachePath:'full-prefill-both-stages',budgetTokens,decodeBudgetMs,stageBudgetMs,actualTokens:reasoning.usage?.completion_tokens??null,reasoningText:draft,injectedPrefix:THINKING_PREFIX,assistantPrefix:'<think>\n'+THINKING_PREFIX,rawText:reasoning.text,messages:reasoningMessages,reasoningClosedNaturally:reasoning.naturalClose,forcedClosure:!reasoning.naturalClose,cutoffReason:reasoning.cutoffReason,durationMs:reasoning.durationMs,firstTokenMs:reasoning.firstTokenMs,finishReason:reasoning.finishReason,usage:reasoning.usage,sampling:{temperature,topP,presencePenalty,topK:null,seed:42}},selection:{messages:finalMessages,durationMs:selection.durationMs,firstTokenMs:selection.firstTokenMs,finishReason:selection.finishReason,cutoffReason:selection.cutoffReason,stageBudgetMs:data.selectionStageMs??30000,usage:selection.usage}}});
    }
  }catch(error){self.postMessage({id,error:error?.message??String(error)});}
};
