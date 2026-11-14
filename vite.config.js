import {defineConfig} from 'vite';
import {mkdir,writeFile} from 'node:fs/promises';
export default defineConfig({
  worker:{format:'es'},
  server:{proxy:{'/api/catan':{target:'http://127.0.0.1:8001',changeOrigin:true}}},
  build:{rollupOptions:{input:{main:'index.html',catan:'catan.html'}}},
  plugins:[{name:'local-benchmark-results',configureServer(server){
    server.middlewares.use('/__catan_results',async(req,res)=>{
      if(req.method!=='POST'||req.headers.origin!==`http://${req.headers.host}`){res.statusCode=403;res.end();return;}
      try{let body='';for await(const chunk of req){body+=chunk;if(body.length>30_000_000)throw new Error('Report too large');}const report=JSON.parse(body);if(report.app!=='TinySettler-Catan'||!Array.isArray(report.results))throw new Error('Invalid report');await mkdir('catan-results',{recursive:true});await writeFile('catan-results/latest.json',JSON.stringify(report,null,2));for(const r of report.results){if(r.finishedAt)await writeFile(`catan-results/${r.id.replace(/[^a-zA-Z0-9_-]/g,'_')}.json`,JSON.stringify(r,null,2));}res.setHeader('Content-Type','application/json');res.end('{"saved":true}');}catch{res.statusCode=400;res.end('{"saved":false}');}
    });
    server.middlewares.use('/__results',async(req,res)=>{
      if(req.method!=='POST' || req.headers.origin!==`http://${req.headers.host}`) {res.statusCode=403;res.end();return;}
      try {
        let body=''; for await(const chunk of req){body+=chunk;if(body.length>2_000_000)throw new Error('Report too large');}
        const report=JSON.parse(body);
        if(report.app!=='TinySettler' || !Array.isArray(report.results))throw new Error('Invalid report');
        await mkdir('test-results',{recursive:true});
        await writeFile('test-results/latest.json',JSON.stringify(report,null,2));
        res.setHeader('Content-Type','application/json');res.end('{"saved":true}');
      }catch{res.statusCode=400;res.end('{"saved":false}');}
    });
  }}]
});
