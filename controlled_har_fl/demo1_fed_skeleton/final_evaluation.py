"""Freeze development selections, then evaluate immutable models on bound test identities.

The freeze command never loads test tensors. Evaluation cannot select checkpoints.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from . import corrected_fl as fl, development_fl as dev
from .prepare_corrected_fl import class_contract

ROOT = Path(__file__).resolve().parents[1]


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_protocol(protocol):
    mapping, names = class_contract(15)
    if protocol['label_names'] != names or protocol['ntu_zero_based_ids'] != list(mapping):
        raise ValueError('Wrong class map')


def allocate(labels, proportions, seed):
    proportions = np.asarray(proportions)
    if proportions.ndim != 2 or not np.isfinite(proportions).all() or np.any(proportions < 0) or not np.allclose(proportions.sum(1), 1):
        raise ValueError('Invalid train-derived client proportions')
    if labels.dtype.kind not in 'iu' or np.any(labels < 0) or np.any(labels >= len(proportions)):
        raise ValueError('Invalid labels')
    rng = np.random.default_rng(seed)
    clients = np.full(len(labels), -1, dtype=np.int64)
    for label, probs in enumerate(proportions):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        raw = probs * len(indices)
        counts = np.floor(raw).astype(int)
        counts[np.argsort(-(raw-counts), kind='stable')[:len(indices)-counts.sum()]] += 1
        for cid, chunk in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
            clients[chunk] = cid
    validate_clients(clients, len(proportions[0]))
    return clients


def validate_clients(clients, count):
    if clients.ndim != 1 or clients.dtype.kind not in 'iu' or set(clients.tolist()) != set(range(count)):
        raise ValueError('Invalid client assignment')


def validate_identities(ids, subjects, train, development):
    if len(set(ids.tolist())) != len(ids) or len(subjects) != len(ids):
        raise ValueError('Duplicate or invalid test identities')
    for shards in (train, development):
        if set(ids) & set(np.concatenate([s.sequence_ids for s in shards])) or set(subjects) & set(np.concatenate([s.subject_ids for s in shards])):
            raise ValueError('Test identity overlap')


def freeze(run, output):
    run = Path(run).resolve()
    contract = json.loads((run/'contract.json').read_text())
    if contract['purpose'] != 'primary':
        raise ValueError('Only prespecified primary runs can enter final evaluation')
    validate_protocol(contract['protocol'])
    if set(contract['methods']) != set(dev.METHODS):
        raise ValueError('Incomplete method panel')
    records = []
    for seed in contract['seeds']:
        initial = set()
        for method in contract['methods']:
            folder = run/f'seed_{seed}'/method
            result = json.loads((folder/'result.json').read_text())
            if len(result['history']) != contract['rounds'] or result['queue_contract_sha256'] != fl.sha256(run/'contract.json'):
                raise ValueError('Incomplete or changed run')
            chosen = max(result['history'], key=lambda r: (r['development']['pooled']['macro_f1_all_declared_classes'],
                         -r['development']['pooled']['cross_entropy'], -r['round']))['round']
            if chosen != result['selected_round']:
                raise ValueError('Selection rule mismatch')
            checkpoint = folder/'best_development.pt'
            if fl.sha256(checkpoint) != result['artifact_sha256'][checkpoint.name]:
                raise ValueError('Changed checkpoint hash')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if state['round'] != chosen or state['method'] != method or state['seed'] != seed:
                raise ValueError('Wrong selected checkpoint')
            initial.add(result['initial_state_sha256'])
            records.append(dict(seed=seed, method=method, selected_round=chosen,
                checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=fl.sha256(checkpoint),
                client_state_sha256=[dev.state_hash(s) for s in state['clients']],
                server_state_sha256=dev.state_hash(state['server']) if state['server'] else None))
        if len(initial) != 1:
            raise ValueError('Unpaired initial weights')
    frozen = dict(schema_version=1, frozen_utc=datetime.now(timezone.utc).isoformat(),
                  run_contract=contract, run_contract_sha256=fl.sha256(run/'contract.json'),
                  checkpoints=records, test_allocation_seed=contract['test_allocation_seed'],
                  test_seen_during_selection=False,
                  evaluator_sha256=fl.sha256(__file__))
    write_new(output, frozen)
    return frozen


def load_selected(record, factory, num_clients):
    path = ROOT/record['checkpoint']
    if fl.sha256(path) != record['checkpoint_sha256']:
        raise ValueError('Changed checkpoint hash')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if [dev.state_hash(s) for s in state['clients']] != record['client_state_sha256']:
        raise ValueError('Changed client or BN state')
    if (dev.state_hash(state['server']) if state['server'] else None) != record['server_state_sha256']:
        raise ValueError('Changed shared state')
    if (state['round'], state['method'], state['seed']) != (record['selected_round'], record['method'], record['seed']):
        raise ValueError('Wrong checkpoint identity')
    method = record['method']
    if len(state['clients']) != (1 if method == 'centralized' else num_clients):
        raise ValueError('Wrong client count')
    states = ([state['server']] if method in ('fedavg','fedprox') else state['clients'])
    models = []
    for saved in states:
        model=factory()
        model.load_state_dict(saved, strict=True)
        models.append(model)
    return models


def score_fixed(models, shards, method, *, device, batch_size, num_classes):
    before = [dev.state_hash(fl.state_copy(m)) for m in models]
    metrics, predictions = dev.evaluate(models, shards, centralized=method in ('centralized','fedavg','fedprox'),
                                        device=device, batch_size=batch_size, num_classes=num_classes)
    after = [dev.state_hash(fl.state_copy(m)) for m in models]
    if before != after:
        raise ValueError('Model/BN state altered during test inference')
    metrics['mean_client_macro_f1'] = float(np.mean([m['macro_f1_all_declared_classes'] for m in metrics['clients']]))
    metrics['worst_client_macro_f1'] = min(m['macro_f1_all_declared_classes'] for m in metrics['clients'])
    matrix=np.asarray(metrics['pooled']['confusion_matrix'])
    metrics['falling_counts'] = dict(tp=int(matrix[0,0]), fn=int(matrix[0].sum()-matrix[0,0]),
                                   fp=int(matrix[:,0].sum()-matrix[0,0]), support=int(matrix[0].sum()))
    return metrics, predictions, before, after


def load_test(data_dir, protocol, train, development):
    repair=ROOT/'experiments/har_fl_repair_20261001'
    binding_path=repair/'legacy_test_binding.json'
    csv_path=repair/'legacy_test_identities.csv'
    test_path=ROOT/'demo1_fed_skeleton/data/fed_medical_2d/centralized/test.npz'
    original=json.loads((ROOT/'BUNDLE_MANIFEST.json').read_text())['files']
    for path in (binding_path,csv_path,test_path):
        if fl.sha256(path) != original[str(path.relative_to(ROOT))]['sha256']:
            raise ValueError('Original test binding hash changed')
    binding=json.loads(binding_path.read_text())
    if binding['status'] != 'exact_match' or fl.sha256(test_path) != binding['legacy_test_npz_sha256'] or fl.sha256(csv_path) != binding['identity_csv_sha256']:
        raise ValueError('Invalid test binding')
    with csv_path.open() as stream:
        rows=list(csv.DictReader(stream))
    with np.load(test_path, allow_pickle=False) as arrays:
        x,y=arrays['x'],arrays['y']
    ids=np.array([r['sequence_id'] for r in rows])
    subjects=np.array([int(r['subject_id']) for r in rows],dtype=np.int64)
    if (len(rows) != binding['samples'] or x.shape != (len(rows),2,100,17,3) or
        not np.isfinite(x).all() or [int(r['index']) for r in rows] != list(range(len(rows))) or
        not np.array_equal(y,np.array([int(r['label']) for r in rows]))):
        raise ValueError('Test tensor/label/identity alignment changed')
    for sid,subject in zip(ids,subjects):
        if int(re.search(r'P(\d{3})',sid).group(1)) != subject:
            raise ValueError('Wrong performer identity')
    if not set(subjects).issubset(protocol['excluded_official_test_subjects']):
        raise ValueError('Wrong test performers')
    validate_identities(ids,subjects,train,development)
    return x,y,ids,subjects


def evaluate(frozen_path, data_dir, output):
    from experiments.har_fl_lab_20261024.gpu_queue import factory, configure
    configure()
    frozen=json.loads(Path(frozen_path).read_text())
    if frozen['evaluator_sha256'] != fl.sha256(__file__):
        raise ValueError('Evaluator changed since selection freeze')
    contract=frozen['run_contract']
    validate_protocol(contract['protocol'])
    for name,digest in contract['source_sha256'].items():
        if fl.sha256(ROOT/name) != digest:
            raise ValueError(f'Changed experiment source: {name}')
    protocol,train,development=dev.load_development(data_dir)
    if protocol != contract['protocol'] or fl.sha256(Path(data_dir)/'protocol.json') != contract['protocol_sha256']:
        raise ValueError('Changed data protocol')
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    # All selected file/state hashes checked before opening test arrays.
    for record in frozen['checkpoints']:
        load_selected(record,factory,protocol['num_clients'])
    x,y,ids,subjects=load_test(data_dir,protocol,train,development)
    clients=allocate(y,protocol['label_client_proportions'],frozen['test_allocation_seed'])
    np.savez_compressed(out/'allocation.npz',y=y,sequence_ids=ids,subject_ids=subjects,client_ids=clients)
    write_new(out/'allocation.json',dict(seed=frozen['test_allocation_seed'],
        scope='Ground-truth-label allocated synthetic cohorts; no deployment router; historically explored test benchmark',
        allocation_sha256=fl.sha256(out/'allocation.npz'),
        client_class_support=[np.bincount(y[clients==c],minlength=15).tolist() for c in range(protocol['num_clients'])],
        frozen_selection_sha256=fl.sha256(frozen_path)))
    shards=[fl.Shard(torch.from_numpy(x[clients==c].astype(np.float32)),torch.from_numpy(y[clients==c].astype(np.int64)),
                     ids[clients==c],subjects[clients==c]) for c in range(protocol['num_clients'])]
    del x,train,development
    results=[]
    for record in frozen['checkpoints']:
        models=load_selected(record,factory,protocol['num_clients'])
        metrics,predictions,before,after=score_fixed(models,shards,record['method'],device='cuda',batch_size=contract['batch_size'],num_classes=15)
        name=f"seed_{record['seed']}_{record['method']}"
        np.savez_compressed(out/f'{name}.npz',**predictions)
        result=dict(**record,metrics=metrics,state_hashes_before=before,state_hashes_after=after,
                    prediction_sha256=fl.sha256(out/f'{name}.npz'))
        write_new(out/f'{name}.json',result)
        results.append(result)
        print(f'EVALUATED {name}',flush=True)
    write_new(out/'results.json',dict(complete=True,results=results,frozen_selection_sha256=fl.sha256(frozen_path)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze'); f.add_argument('--run',type=Path,required=True); f.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('evaluate'); e.add_argument('--frozen',type=Path,required=True); e.add_argument('--data-dir',type=Path,required=True); e.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='freeze':freeze(a.run,a.output)
    else:evaluate(a.frozen,a.data_dir,a.output)


if __name__=='__main__':main()
