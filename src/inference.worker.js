import {CreateMLCEngine,prebuiltAppConfig} from '@mlc-ai/web-llm';
import {actionSchema} from './action-schema.js';
let engine;
self.onmessage=async({data})=>{
  const {id,type}=data;
  try {
    if(type==='load') {
      const record=prebuiltAppConfig.model_list.find(m=>m.model_id===data.model);
      engine=await CreateMLCEngine(data.model,{initProgressCallback:progress=>self.postMessage({type:'progress',progress})},{context_window_size:8192,conv_config:{...record?.overrides?.conv_config,role_empty_sep:'\n<think>\n'}});
      self.postMessage({id,result:{vendor:await engine.getGPUVendor()}});
    } else if(type==='reset') {
      await engine.resetChat(); self.postMessage({id,result:true});
    } else if(type==='generate') {
      if(data.thinking) await engine.resetChat();
      const start=performance.now(); let text='',firstTokenMs=null,usage,finishReason;
      const stream=await engine.chat.completions.create({messages:data.messages,temperature:data.temperature??0,top_p:data.topP??1,presence_penalty:data.presencePenalty??0,max_tokens:data.maxTokens??160,seed:42,stream:true,stream_options:{include_usage:true},extra_body:{enable_thinking:!!data.thinking},...(!data.thinking&&data.outputMode==='schema'?{response_format:{type:'json_object',schema:JSON.stringify(data.schema??actionSchema)}}:{})});
      for await(const chunk of stream) {
        const part=chunk.choices?.[0]?.delta?.content;
        if(part) { firstTokenMs??=performance.now()-start; text+=part; self.postMessage({type:'token',text:data.thinking?'<think>\n'+text:text}); }
        if(chunk.usage) usage=chunk.usage;
        if(chunk.choices?.[0]?.finish_reason) finishReason=chunk.choices[0].finish_reason;
      }
      self.postMessage({id,result:{text:data.thinking?'<think>\n'+text:text,firstTokenMs,durationMs:performance.now()-start,usage,finishReason}});
    }
  } catch(error) {self.postMessage({id,error:error?.message??String(error)});}
};
