"""Exercise STOP_REQUESTED and durable resume through the real CUDA queue on pilot data."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from demo1_fed_skeleton import development_fl as dev
    out=ROOT/'lab_results/queue_resume_check'
    out.mkdir(exist_ok=False)
    cmd=[sys.executable,str(ROOT/'experiments/har_fl_lab_20261024/gpu_queue.py'),
         '--data-dir',str(ROOT/'experiments/har_fl_repair_20261001/pilot_data'),
         '--output-dir',str(out),'--rounds','3','--seeds','42','--methods','fedbn',
         '--batch-size','16','--purpose','pilot',
         '--deadline',(datetime.now(timezone.utc)+timedelta(minutes=10)).isoformat()]
    folder=out/'seed_42/fedbn'
    with (out/'execution.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        requested=False
        while proc.poll() is None:
            pointer=folder/'resume.json'
            if not requested and pointer.exists():
                record=json.loads(pointer.read_text())
                if record['file']!='resume_000.pt':
                    (out/'STOP_REQUESTED').touch();requested=True
            time.sleep(.02)
        assert proc.returncode==0 and requested
        assert json.loads((out/'status.json').read_text())['state']=='stopped_resumable'
        saved=json.loads((folder/'resume.json').read_text())
        state=torch.load(folder/saved['file'],map_location='cpu',weights_only=True)
        completed=state['current']['round'];assert 1<=completed<3
        (out/'STOP_REQUESTED').unlink()
        subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    result=json.loads((folder/'result.json').read_text())
    assert [h['round'] for h in result['history']]==[1,2,3]
    assert result['training_examples_processed']==900
    report=dict(passed=True,interrupted_after_round=completed,resumed_to_round=3,
                examples=result['training_examples_processed'],scope='Actual backbone, pilot data, CUDA queue STOP_REQUESTED and disk resume; exact numerical resume separately checked in cuda_resume.json')
    (ROOT/'lab_results/queue_resume.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
