"""Linux CPU utilization and trainer RSS samples; no training-state access."""
import argparse,csv,os,time
from pathlib import Path

def counters(pid):
    cpu=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    rss=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    return sum(cpu),cpu[3]+cpu[4],int(stat[11])+int(stat[12]),rss

def main(a):
    hz=os.sysconf('SC_CLK_TCK'); prev=counters(a.pid); last=time.monotonic()
    with a.output.open('w',buffering=1) as f:
        w=csv.writer(f); w.writerow(['unix_time','system_cpu_pct','trainer_cpu_pct_one_core','trainer_cpu_pct_capacity','trainer_rss_bytes'])
        while time.time()<a.deadline:
            time.sleep(10)
            try: now=counters(a.pid)
            except FileNotFoundError: break
            ts=time.monotonic(); delta=ts-last; total=now[0]-prev[0]
            percent=100*(now[2]-prev[2])/hz/delta
            w.writerow([time.time(),100*(1-(now[1]-prev[1])/max(1,total)),percent,percent/os.cpu_count(),now[3]])
            prev=now; last=ts
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--pid',type=int,required=True); p.add_argument('--deadline',type=float,required=True); p.add_argument('--output',type=Path,required=True); main(p.parse_args())
