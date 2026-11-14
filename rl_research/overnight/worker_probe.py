"""Measure one local engine against two synchronous subprocess engines.
The latter includes observation/mask serialization, as a central GPU actor needs.
"""
import multiprocessing as mp
import time
import numpy as np
from env import Env

def worker(conn,seed):
    e=Env(seed)
    try:
        while True:
            a=conn.recv()
            if a is None: break
            _,_,term,trunc,_=e.step(a)
            if term or trunc: seed+=1; e=Env(seed)
            conn.send((e.obs(),e.mask()))
    finally: conn.close()

def probe(decisions=2048):
    start=time.perf_counter(); e=Env(88231)
    for i in range(decisions):
        _,_,term,trunc,_=e.step(int(np.flatnonzero(e.mask())[i%len(np.flatnonzero(e.mask()))]))
        e.obs(); e.mask()
        if term or trunc: e=Env(88232+i)
    single=decisions/(time.perf_counter()-start)
    ctx=mp.get_context('spawn'); pairs=[ctx.Pipe() for _ in range(2)]
    procs=[ctx.Process(target=worker,args=(child,88231+j)) for j,(_,child) in enumerate(pairs)]
    masks=[Env(88231+j).mask() for j in range(2)]
    for p in procs: p.start()
    # Warm startup excluded, but each decision includes central actor transport.
    for j,(parent,_) in enumerate(pairs): parent.send(int(np.flatnonzero(masks[j])[0]))
    for j,(parent,_) in enumerate(pairs): _,masks[j]=parent.recv()
    start=time.perf_counter()
    for i in range(decisions//2):
        for j,(parent,_) in enumerate(pairs):
            legal=np.flatnonzero(masks[j]); parent.send(int(legal[i%len(legal)]))
        for j,(parent,_) in enumerate(pairs): _,masks[j]=parent.recv()
    parallel=decisions/(time.perf_counter()-start)
    for parent,_ in pairs: parent.send(None)
    for p in procs: p.join(10)
    return dict(local_one_decisions_s=single,subprocess_two_decisions_s=parallel,selected_workers=1,reason='single engine avoids synchronous IPC and retains exact sequential learner transitions; two-worker transport comparison recorded')
if __name__=='__main__': print(probe())
