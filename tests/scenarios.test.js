import {test} from 'node:test';
import assert from 'node:assert/strict';
import {scenarios,parseAction,createWorld,runScenario,answerOnly,promptForMode} from '../src/scenarios.js';
import {formatQwenChat} from '../src/chat-format.js';
test('fabricated finals do not pass tool objectives',()=>{for(const s of scenarios)assert.equal(createWorld(s).grade(String(s.expected??'Done')),false);});
test('craft requires and consumes resources',()=>{const w=createWorld(scenarios[2]);assert.ok(w.execute({tool:'craft',args:{item:'shelter'}}).error);w.execute({tool:'gather',args:{resource:'wood',amount:4}});w.execute({tool:'gather',args:{resource:'stone',amount:2}});w.execute({tool:'craft',args:{item:'shelter'}});assert.equal(w.grade('Done'),true);assert.equal(w.state.wood,0);assert.equal(w.state.stone,0);});
test('recovery must actually encounter the requested failure',()=>{const w=createWorld(scenarios[4]);w.execute({tool:'deliver',args:{parcel:'A',route:'ferry'}});assert.equal(w.grade('Done'),false);const good=createWorld(scenarios[4]);assert.ok(good.execute({tool:'deliver',args:{parcel:'A',route:'bridge'}}).error);good.execute({tool:'deliver',args:{parcel:'A',route:'ferry'}});assert.equal(good.grade('Done'),true);});
test('invalid input cannot mutate world properties',()=>{const w=createWorld(scenarios[2]);for(const resource of ['__proto__','constructor','wood'])assert.ok(w.execute({tool:'gather',args:{resource,amount:-1}}).error);assert.equal(w.state.wood,0);assert.throws(()=>parseAction('{"tool":"inventory","args":{},"final":"done"}'));});
test('runner observes feedback and grades final answer',async()=>{const replies=['oops','{"tool":"calculate","args":{"op":"multiply","a":7,"b":13}}','{"tool":"calculate","args":{"op":"add","a":91,"b":9}}','{"final":"100"}'];let n=0;const result=await runScenario(scenarios[0],async messages=>{if(n===1)assert.match(messages.at(-1).content,/Invalid action/);return {text:replies[n++]};});assert.equal(result.passed,true);assert.equal(result.trace[0].parseError,true);assert.equal(result.trace.length,4);});
test('step limit is a failure',async()=>{const r=await runScenario(scenarios[0],async()=>({text:'{"tool":"inventory","args":{}}'}),()=>{},2);assert.equal(r.passed,false);assert.equal(r.reason,'Step limit reached');});
test('thinking is parsed only after closing tag and is removed from history',async()=>{assert.equal(answerOnly('<think>unfinished'), '');assert.throws(()=>parseAction('<think>{"final":"100"}'));assert.equal(parseAction('<think>private analysis</think>{"final":"100"}').final,'100');let n=0;const r=await runScenario(scenarios[0],async messages=>{if(n++) {assert.equal(messages[2].content,'{"tool":"calculate","args":{"op":"add","a":91,"b":9}}');return {text:'<think>Done</think>{"final":"100"}'};} assert.equal(messages[0].content,promptForMode(true));return {text:'<think>Plan</think>{"tool":"calculate","args":{"op":"add","a":91,"b":9}}'};},()=>{},10,{thinking:true});assert.equal(r.passed,true);});
test('Qwen template opens or closes thinking without altering history',()=>{const messages=[{role:'system',content:'Rules'},{role:'user',content:'Task'}];assert.equal(formatQwenChat(messages,true),'<|im_start|>system\nRules<|im_end|>\n<|im_start|>user\nTask<|im_end|>\n<|im_start|>assistant\n<think>\n');assert.ok(formatQwenChat(messages,false).endsWith('<think>\n\n</think>\n\n'));});
test('a thinking cutoff fails the episode without repeating the same incomplete thought',async()=>{let calls=0;const r=await runScenario(scenarios[0],async()=>{calls++;return {text:'<think>unfinished',finishReason:'length'};},()=>{},10,{thinking:true});assert.equal(calls,1);assert.equal(r.passed,false);assert.equal(r.reason,'Thinking/output token limit reached');});

const {parseNativeAction}=await import('../src/native-tools.js');
test('native tool calls reject multiple calls and malformed parameters',()=>{
  const xml='<tool_call><function=calculate><parameter=op>add</parameter><parameter=a>91</parameter><parameter=b>9</parameter></function></tool_call>';
  assert.deepEqual(parseNativeAction(xml),{tool:'calculate',args:{op:'add',a:91,b:9}});
  assert.throws(()=>parseNativeAction(xml+xml));
  assert.throws(()=>parseNativeAction(xml.replace('>91<','>invalid<')));
  assert.throws(()=>parseNativeAction(xml.replace('<parameter=b>9</parameter>','')));
});
test('native runner uses tool observations and still requires actual tool use',async()=>{
  let calls=0;
  const r=await runScenario(scenarios[0],async messages=>{
    if(calls++){assert.equal(messages.at(-1).role,'tool');assert.equal(messages.at(-1).content,'{"value":100}');return {text:'100'};}
    return {text:'<tool_call><function=calculate><parameter=op>add</parameter><parameter=a>91</parameter><parameter=b>9</parameter></function></tool_call>'};
  },()=>{},10,{protocol:'native'});
  assert.equal(r.passed,true);
  const fake=await runScenario(scenarios[0],async()=>({text:'100'}),()=>{},10,{protocol:'native'});assert.equal(fake.passed,false);
});
