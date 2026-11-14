import json,time
from train import Trainer
from remote import resume_training

def test_runner_resumes_with_original_deadline(tmp_path):
    t=Trainer('ppo',128,71,4096)
    for _ in range(32): t.collect()
    t.elapsed=100
    t.save(tmp_path/'2048.pt') # filename is planned checkpoint; counter is authoritative
    restored,stop=resume_training(tmp_path,'ppo',128,71,4096,'cpu',time.time()+10000)
    assert restored.n==32 and len(restored.rollout)==32
    assert 2590<stop-time.time()<=2600
    restored.collect(); restored.save(tmp_path/'2048.pt')
    again,second_stop=resume_training(tmp_path,'ppo',128,71,4096,'cpu',time.time()+10000)
    assert again.n==33 and stop==second_stop
    (tmp_path/'run-budget.json').write_text(json.dumps({'deadline':time.time()-1}))
    _,expired=resume_training(tmp_path,'ppo',128,71,4096,'cpu',time.time()+10000)
    assert expired<time.time()

def test_chunked_upload_reassembly(tmp_path):
    import shutil
    from controller import upload_large
    source=tmp_path/'source.gz'; source.write_bytes(b'x'*(8*1024*1024)+b'last-block')
    destination=tmp_path/'remote.gz'; sent=[]
    def cli(*args,**kwargs):
        assert args[:3]==('upload','-s','test-session')
        shutil.copy2(args[3],args[4]); sent.append(args[4])
    def remote(code,**kwargs): exec(compile(code,'<reassembly-test>','exec'),{})
    upload_large(source,str(destination),tmp_path,'test-session',cli,remote)
    assert len(sent)==2 and destination.read_bytes()==source.read_bytes()
