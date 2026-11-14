// Match the exact text-only ChatML envelope used by the WebLLM control.
// Keep template differences out of the ONNX-vs-MLC comparison.
export function assistantPrefix(thinking){return thinking?'<think>\n':'<think>\n\n</think>\n\n';}
export function formatQwenChat(messages,thinking=false){
  return messages.map(({role,content})=>`<|im_start|>${role}\n${content}<|im_end|>\n`).join('')+'<|im_start|>assistant\n'+assistantPrefix(thinking);
}
