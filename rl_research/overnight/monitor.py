"""Optional ten-second runtime telemetry, independent of training RNG."""
import csv,os,resource,subprocess,time
from pathlib import Path

def sample(path,deadline):
    with open(path,'w',buffering=1) as f:
        writer=csv.writer(f); writer.writerow(['unix_time','load1','load5','cpu_count','gpu_csv'])
        while time.time()<deadline:
            result=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,capture_output=True)
            writer.writerow([time.time(),*os.getloadavg()[:2],os.cpu_count(),result.stdout.strip()])
            time.sleep(10)
if __name__=='__main__':
    import sys
    sample(sys.argv[1],float(sys.argv[2]))
