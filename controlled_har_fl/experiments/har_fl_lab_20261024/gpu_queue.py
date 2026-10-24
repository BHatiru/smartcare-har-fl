"""Bounded CUDA-only queue with immutable contracts and atomic round resume."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.har_fl_repair_20261001.run_background_fl import atomic_json, atomic_checkpoint, acquire_lock, StopRequested


def compact_checkpoint(value):
    """Share byte-identical immutable CPU tensors for torch serialization only."""
    import torch
    cache={}
    def visit(item):
        if isinstance(item,torch.Tensor):
            tensor=item.detach().cpu().contiguous()
            key=(str(tensor.dtype),tuple(tensor.shape),hashlib.sha256(tensor.numpy().tobytes()).digest())
            if key not in cache:cache[key]=tensor
            elif not torch.equal(cache[key],tensor):raise ValueError('Tensor digest collision')
            return cache[key]
        if isinstance(item,dict):return {k:visit(v) for k,v in item.items()}
        if isinstance(item,list):return [visit(v) for v in item]
        if isinstance(item,tuple):return tuple(visit(v) for v in item)
        return item
    return visit(value)


def source_hashes():
    from demo1_fed_skeleton.corrected_fl import sha256
    paths = list((ROOT / 'demo1_fed_skeleton').rglob('*.py'))
    paths += list(Path(__file__).parent.glob('*.py'))
    paths.append(ROOT/'experiments/har_fl_repair_20261001/run_background_fl.py')
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def configure():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback forbidden')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def factory():
    from demo1_fed_skeleton.models.stgcn import build_model
    return build_model(num_classes=15, in_channels=3, graph='coco', base_channels=64,
                       num_stages=6, num_person=2, dropout=.3)


def main():
    import numpy as np
    import torch
    from demo1_fed_skeleton import corrected_fl as fl, development_fl as dev
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--rounds', type=int, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--methods', nargs='+', choices=dev.METHODS, default=list(dev.METHODS))
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--deadline', required=True, help='Absolute ISO8601 timestamp including timezone')
    p.add_argument('--purpose', choices=['pilot', 'timing', 'primary'], required=True)
    a = p.parse_args()
    deadline = datetime.fromisoformat(a.deadline).timestamp()
    if min(a.rounds, a.batch_size) < 1 or len(set(a.seeds)) != len(a.seeds) or len(set(a.methods)) != len(a.methods):
        raise ValueError('Invalid budget or repeated jobs')
    configure()
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(out / 'RUN.lock')
    protocol, train, development = dev.load_development(a.data_dir)
    binding = dict(schema_version=1, source_sha256=source_hashes(), protocol_sha256=fl.sha256(a.data_dir/'protocol.json'),
                   protocol=protocol, rounds=a.rounds, seeds=a.seeds, methods=a.methods,
                   batch_size=a.batch_size, lr=.01, local_epochs=1, proximal_mu=.01,
                   schedule='base R1-20, 0.1x R21-35, 0.01x R36+', device='cuda', precision='float32, TF32 disabled',
                   torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version(),
                   gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                   driver=os.popen('nvidia-smi --query-gpu=driver_version --format=csv,noheader').read().strip(),
                   purpose=a.purpose, deadline=a.deadline,
                   test_allocation_seed=20261024,
                   selection='Maximum pooled development macro-F1; lower cross-entropy; earliest round',
                   optimizer='SGD momentum=0.9 weight_decay=0.0005 reset each round',
                   pooled_bn_calibration=False, warm_start=False)
    contract = out/'contract.json'
    if contract.exists():
        if json.loads(contract.read_text()) != binding:
            raise ValueError('Immutable source/data/environment/schedule/deadline contract changed')
    else:
        atomic_json(contract, binding)
    for name,digest in binding['source_sha256'].items():
        snapshot=out/'source_snapshot'/name
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        if not snapshot.exists():shutil.copy2(ROOT/name,snapshot)
        if fl.sha256(snapshot)!=digest:
            raise ValueError('Source snapshot hash mismatch')
    interrupted = False
    def stop_signal(*_):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    current = {'state': 'ready'}
    def status(**kw):
        current.update(kw, updated_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
        atomic_json(out/'status.json', current)
        if a.purpose=='primary':
            text=(f"# HAR/FL lab status\n\nStage: {current['state']}.\n\n"
                  f"Seed: {current.get('seed')}; method: {current.get('method')}; "
                  f"completed round: {current.get('round',0)} / {a.rounds}.\n"
                  f"Updated UTC: {current['updated_utc']}. Absolute training deadline: {a.deadline}.\n\n"
                  "Evidence: primary/contract.json, primary/status.json, primary.log and per-method round/checkpoint files.\n"
                  "Development monitoring only; final test scoring starts after all selections are frozen.\n")
            temporary=ROOT/'lab_results/STATUS.md.tmp'
            temporary.write_text(text)
            os.replace(temporary,ROOT/'lab_results/STATUS.md')
    def check_stop():
        if interrupted or (out/'STOP_REQUESTED').exists() or time.time() >= deadline:
            raise StopRequested('Signal, STOP_REQUESTED, or absolute deadline reached')
        if shutil.disk_usage(out).free < 500_000_000:
            raise StopRequested('Less than 500 MB free; preserving completed work')
    try:
        for seed in a.seeds:
            for method in a.methods:
                check_stop()
                folder = out/f'seed_{seed}'/method
                folder.mkdir(parents=True, exist_ok=True)
                if (folder/'result.json').exists():
                    result = json.loads((folder/'result.json').read_text())
                    for name, digest in result['artifact_sha256'].items():
                        if fl.sha256(folder/name) != digest:
                            raise ValueError(f'Changed artifact {folder/name}')
                    continue
                status(state='training', seed=seed, method=method, round=0)
                torch.cuda.reset_peak_memory_stats()
                pointer = folder/'resume.json'
                resume = None
                if pointer.exists():
                    saved = json.loads(pointer.read_text())
                    latest = folder/saved['file']
                    if fl.sha256(latest) != saved['sha256']:
                        raise ValueError('Changed resume checkpoint')
                    resume = torch.load(latest, map_location='cpu', weights_only=True)
                if resume and resume['queue_contract_sha256'] != fl.sha256(contract):
                    raise ValueError('Resume contract changed')
                if resume:
                    for entry in resume['history']:
                        atomic_json(folder/f"round_{entry['round']:03d}.json",entry)
                def save(state):
                    state['queue_contract_sha256'] = fl.sha256(contract)
                    r = state['current']['round']
                    latest=folder/f'resume_{r:03d}.pt'
                    atomic_checkpoint(torch, latest, compact_checkpoint(state))
                    atomic_json(pointer,dict(file=latest.name,sha256=fl.sha256(latest)))
                    # Pointer commits the durable checkpoint before old resume storage is reclaimed.
                    for old in folder.glob('resume_*.pt'):
                        if old != latest:old.unlink()
                    if r:
                        atomic_json(folder/f'round_{r:03d}.json', state['history'][-1])
                        for tag, predictions in [('selected', state['best_predictions']), ('final', state['last_predictions'])]:
                            temp = folder/f'{tag}.tmp.npz'
                            np.savez_compressed(temp, **dev.unpack_predictions(predictions))
                            os.replace(temp, folder/f'{tag}_development.npz')
                    status(round=r, selected_round=state['best_checkpoint']['round'] if r else None)
                last_log = 0.
                def progress(entry):
                    nonlocal last_log
                    check_stop()
                    if time.monotonic()-last_log >= 30:
                        status(progress=entry)
                        print(json.dumps(dict(seed=seed, method=method, **entry)), flush=True)
                        last_log=time.monotonic()
                def round_done(entry):
                    print(json.dumps({'seed':seed,'method':method,'round':entry['round'],
                                      'seconds':entry['seconds'],'development':entry['development']['pooled']}), flush=True)
                    check_stop()
                result, endpoints = dev.run_method(factory, train, development, method=method, rounds=a.rounds,
                    local_epochs=1, batch_size=a.batch_size, lr=.01, seed=seed, device='cuda', num_classes=15,
                    proximal_mu=.01, resume_state=resume, on_checkpoint=save, on_progress=progress, on_round=round_done)
                artifacts = {}
                for endpoint, (predictions, checkpoint) in endpoints.items():
                    checkpoint['queue_contract_sha256'] = fl.sha256(contract)
                    checkpoint['protocol'] = protocol
                    atomic_checkpoint(torch, folder/f'{endpoint}.pt', compact_checkpoint(checkpoint))
                    np.savez_compressed(folder/f'{endpoint}.npz', **predictions)
                    for ext in ('pt','npz'):
                        name=f'{endpoint}.{ext}'
                        artifacts[name]=fl.sha256(folder/name)
                result.update(artifact_sha256=artifacts, queue_contract_sha256=fl.sha256(contract), test_evaluated=False,
                              gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                atomic_json(folder/'result.json', result)
                del resume, result, endpoints
                gc.collect()
                torch.cuda.empty_cache()
        status(state='complete')
    except StopRequested as e:
        status(state='stopped_resumable', detail=str(e))
    except BaseException as e:
        status(state='failed_resumable', detail=repr(e))
        traceback.print_exc()
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
