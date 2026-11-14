"""Deterministic telemetry, no model involvement."""
import csv,json,os,subprocess,time,sys
from pathlib import Path
out=Path(sys.argv[1]);deadline=float(sys.argv[2]);last={};ticks=os.sysconf('SC_CLK_TCK');previous=time.time()
with (out/'telemetry.csv').open('a') as f:
 w=csv.DictWriter(f,fieldnames=['time','gpu','active_pids','cpu_pct_one_core','rss_bytes','cpu_count']);
 if f.tell()==0:w.writeheader()
 while time.time()<deadline-60:
  try:
   ids=json.loads((out/'active-pids.json').read_text());now=time.time();cpu=0;rss=0
   for pid in ids:
    try:
     stat=Path(f'/proc/{pid}/stat').read_text().split();value=(int(stat[13])+int(stat[14]))/ticks;cpu+=max(0,value-last.get(pid,value))/(now-previous)*100;rss+=int(stat[23])*os.sysconf('SC_PAGE_SIZE');last[pid]=value
    except (FileNotFoundError,ProcessLookupError):pass
   gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
   w.writerow(dict(time=now,gpu=gpu,active_pids=ids,cpu_pct_one_core=cpu,rss_bytes=rss,cpu_count=os.cpu_count()));f.flush();previous=now
  except Exception:pass
  time.sleep(10)
