import {nativeTools} from './native-tools.js';
import {AutoTokenizer,AutoModelForCausalLM,TextStreamer,env} from '@huggingface/transformers';
import {formatQwenChat,assistantPrefix} from './chat-format.js';
env.allowLocalModels=false;
env.backends.onnx.wasm.numThreads=1;
let model,tokenizer;
const modelId='onnx-community/Qwen3.5-2B-ONNX-OPT';
self.onmessage=async({data})=>{
  const {id,type}=data;
  try{
    if(type==='load'){
      tokenizer=await AutoTokenizer.from_pretrained(modelId);
      model=await AutoModelForCausalLM.from_pretrained(modelId,{device:'webgpu',dtype:data.model.split(':')[1],progress_callback:p=>self.postMessage({type:'progress',progress:{progress:(p.progress??0)/100,text:`${p.status} ${p.file??''} ${p.progress?.toFixed(0)??''}%`}})});
      self.postMessage({id,result:{model:modelId}});
    }else if(type==='reset')self.postMessage({id,result:true});
    else if(type==='generate'){
      const start=performance.now();let text='',firstTokenMs=null,firstTokenAt=null,lastTokenAt=null,tokenCount=0;
      const prompt=data.protocol==='native'?tokenizer.apply_chat_template(data.messages,{tools:nativeTools,tokenize:false,add_generation_prompt:true,enable_thinking:!!data.thinking}):formatQwenChat(data.messages,!!data.thinking);
      const inputs=tokenizer(prompt,{add_special_tokens:false});
      const streamer=new TextStreamer(tokenizer,{skip_prompt:true,skip_special_tokens:false,token_callback_function:tokens=>{firstTokenAt??=performance.now();firstTokenMs??=firstTokenAt-start;lastTokenAt=performance.now();tokenCount+=tokens.length;},callback_function:part=>{if(part){firstTokenMs??=performance.now()-start;text+=part;self.postMessage({type:'token',text:data.thinking?'<think>\n'+text:text});}}});
      const outputs=await model.generate({...inputs,max_new_tokens:data.maxTokens??160,do_sample:(data.temperature??0)>0,temperature:data.temperature||1,top_p:data.topP??1,top_k:20,streamer,return_dict_in_generate:true});
      const ids=outputs.sequences.tolist()[0].slice(inputs.input_ids.dims[1]);
      const decoded=tokenizer.decode(ids,{skip_special_tokens:false}).replace(/<\|im_end\|>\s*$/,'').replace(/<\|endoftext\|>\s*$/,'');
      const durationMs=performance.now()-start;
      self.postMessage({id,result:{text:assistantPrefix(!!data.thinking)+decoded,firstTokenMs,durationMs,usage:{completion_tokens:ids.length,prompt_tokens:inputs.input_ids.dims[1],extra:{decode_tokens_per_s:tokenCount>1&&lastTokenAt>firstTokenAt?(tokenCount-1)/((lastTokenAt-firstTokenAt)/1000):null}},finishReason:ids.length>=data.maxTokens?'length':'stop'}});
      for(const t of Object.values(outputs.past_key_values??{}))t?.dispose?.();
    }
  }catch(error){self.postMessage({id,error:error?.message??String(error)});}
};
