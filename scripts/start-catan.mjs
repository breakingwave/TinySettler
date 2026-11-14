import {spawn} from 'node:child_process';
import {existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));
const python=root+'.venv/bin/python';
if(!existsSync(python)){console.error('Create .venv and install backend/requirements-lock.txt first. See backend/README.md.');process.exit(1);}
let stopping=false;const children=[];
function stop(code=0){if(stopping)return;stopping=true;for(const child of children)child.kill('SIGTERM');process.exitCode=code;}
for(const [command,args,env] of [[python,['backend/catan_server.py'],{...process.env,PYTHONHASHSEED:'0'}],[process.execPath,['node_modules/vite/bin/vite.js','--host','127.0.0.1','--port','5173','--strictPort'],process.env]]){const child=spawn(command,args,{cwd:root,env,stdio:'inherit'});children.push(child);child.on('error',e=>{console.error(e.message);stop(1);});child.on('exit',code=>{if(!stopping)stop(code??1);});}
process.on('SIGINT',()=>stop());process.on('SIGTERM',()=>stop());
console.log('Catan agents: http://127.0.0.1:5173/catan.html');
