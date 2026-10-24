"""Independently reconstruct confusion metrics and inspect saved experiment evidence."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from demo1_fed_skeleton import corrected_fl as fl, development_fl as dev


def independent_metrics(predictions, classes=15):
    y=predictions['y']; scores=predictions['logits']
    if scores.shape != (len(y),classes) or not np.isfinite(scores).all():
        raise AssertionError('Invalid scores')
    predicted=scores.argmax(1)
    matrix=np.bincount(y*classes+predicted,minlength=classes*classes).reshape(classes,classes)
    support=matrix.sum(1); guessed=matrix.sum(0); tp=np.diag(matrix)
    f1=np.divide(2*tp,support+guessed,out=np.zeros(classes),where=(support+guessed)>0)
    return dict(n=len(y),accuracy=float(np.trace(matrix)/len(y)),macro_f1_all_declared_classes=float(f1.mean()),
                confusion_matrix=matrix.tolist())


def check_predictions(path, expected, by_id, checks):
    with np.load(path,allow_pickle=False) as arrays:p=dict(arrays)
    ids=p['sequence_ids'].tolist()
    assert len(ids)==len(set(ids))==len(by_id) and set(ids)==set(by_id), 'Missing/duplicate prediction identity'
    assert all(by_id[s]==(int(y),int(subject),int(client)) for s,y,subject,client in
               zip(ids,p['y'],p['subject_ids'],p['client_ids'])), 'Misbound predictions'
    actual=independent_metrics(p)
    for key,value in actual.items():
        assert value==expected['pooled'][key], f'Metric mismatch {path} {key}'
    for cid,metrics in enumerate(expected['clients']):
        mask=p['client_ids']==cid
        actual=independent_metrics({k:v[mask] for k,v in p.items()})
        for key,value in actual.items():assert value==metrics[key],f'Client metric mismatch {path} {cid} {key}'
    checks.append(str(path.relative_to(ROOT)))
    return p


def audit(run,data_dir,test=None):
    run=Path(run).resolve()
    contract=json.loads((run/'contract.json').read_text())
    protocol,train,development=dev.load_development(data_dir)
    assert protocol==contract['protocol']
    for name,digest in contract['source_sha256'].items():
        assert fl.sha256(run/'source_snapshot'/name)==digest,'Producing source snapshot changed'
    by_id={str(s):(int(y),int(subject),cid) for cid,shard in enumerate(development)
           for s,y,subject in zip(shard.sequence_ids,shard.y,shard.subject_ids)}
    checks=[]; rows=[]; incomplete=[]
    from experiments.har_fl_lab_20261024.gpu_queue import factory,configure
    configure()
    for seed in contract['seeds']:
        initial=set()
        for method in contract['methods']:
            folder=run/f'seed_{seed}'/method
            if not (folder/'result.json').exists():
                incomplete.append(dict(seed=seed,method=method));continue
            result=json.loads((folder/'result.json').read_text())
            initial.add(result['initial_state_sha256'])
            assert result['training_examples_processed']==sum(map(len,train))*contract['rounds']
            steps=math.ceil(sum(map(len,train))/contract['batch_size']) if method=='centralized' else sum(math.ceil(len(s)/contract['batch_size']) for s in train)
            assert result['optimizer_steps']==steps*contract['rounds']
            chosen=max(result['history'],key=lambda r:(r['development']['pooled']['macro_f1_all_declared_classes'],
                        -r['development']['pooled']['cross_entropy'],-r['round']))['round']
            assert chosen==result['selected_round']
            for name,digest in result['artifact_sha256'].items():assert fl.sha256(folder/name)==digest
            for endpoint,metrics in [('best_development',result['best_development']),('final',result['final_development'])]:
                predictions=check_predictions(folder/f'{endpoint}.npz',metrics,by_id,checks)
                checkpoint=torch.load(folder/f'{endpoint}.pt',map_location='cpu',weights_only=True)
                assert checkpoint['round']==(chosen if endpoint=='best_development' else contract['rounds'])
                if endpoint=='best_development':
                    # Same GPU and original batch size; replay the whole selected prediction set.
                    models=[]
                    for state in checkpoint['clients']:
                        model=factory();model.load_state_dict(state);models.append(model)
                    _,replayed=dev.evaluate(models,development,centralized=method=='centralized',device='cuda',
                                           batch_size=contract['batch_size'],num_classes=15)
                    np.testing.assert_array_equal(predictions['logits'],replayed['logits'])
                    checks.append(f'{seed}/{method}: exact full selected-development CUDA replay')
                if method=='fedbn' and endpoint=='final':
                    keys=fl.batchnorm_keys(factory())
                    for cid,state in enumerate(checkpoint['clients']):
                        for key in keys:
                            if key.endswith('num_batches_tracked'):
                                assert state[key].item()==math.ceil(len(train[cid])/contract['batch_size'])*contract['rounds']
            rows.append(dict(seed=seed,method=method,selected_round=chosen,examples=result['training_examples_processed'],
                             optimizer_steps=result['optimizer_steps'],seconds=result['total_seconds'],
                             development_curve=[dict(round=h['round'],lr=h['lr'],
                                 accuracy=h['development']['pooled']['accuracy'],
                                 macro_f1=h['development']['pooled']['macro_f1_all_declared_classes'],
                                 cross_entropy=h['development']['pooled']['cross_entropy']) for h in result['history']],
                             best_minus_final_development_macro_f1=result['best_development']['pooled']['macro_f1_all_declared_classes']-result['final_development']['pooled']['macro_f1_all_declared_classes'],
                             convergence_claim=False))
        assert len(initial)<=1,'Unpaired initialization'
    summary={}
    if test:
        test=Path(test).resolve(); report=json.loads((test/'results.json').read_text())
        with np.load(test/'allocation.npz') as allocation:
            test_ids={str(s):(int(y),int(subject),int(cid)) for s,y,subject,cid in zip(allocation['sequence_ids'],allocation['y'],allocation['subject_ids'],allocation['client_ids'])}
        for result in report['results']:
            name=f"seed_{result['seed']}_{result['method']}"
            path=test/f'{name}.npz'
            assert fl.sha256(path)==result['prediction_sha256']
            assert result['state_hashes_before']==result['state_hashes_after']
            check_predictions(path,result['metrics'],test_ids,checks)
        for method in contract['methods']:
            method_rows=[r for r in report['results'] if r['method']==method]
            summary[method]={key:dict(values=[dict(seed=r['seed'],value=r['metrics']['pooled'][key]) for r in method_rows],
                mean=float(np.mean([r['metrics']['pooled'][key] for r in method_rows])),
                sample_std=float(np.std([r['metrics']['pooled'][key] for r in method_rows],ddof=1)) if len(method_rows)>1 else None)
                for key in ('accuracy','macro_f1_all_declared_classes')}
        paired={}
        lookup={(r['seed'],r['method']):r for r in report['results']}
        for method in contract['methods']:
            if method=='fedavg':continue
            paired[method+'_minus_fedavg']={key:[dict(seed=seed,difference=lookup[seed,method]['metrics']['pooled'][key]-lookup[seed,'fedavg']['metrics']['pooled'][key])
                for seed in contract['seeds']] for key in ('accuracy','macro_f1_all_declared_classes')}
        summary['paired_differences']=paired
    return dict(passed=True,complete=not incomplete,checks=checks,accounting=rows,incomplete=incomplete,test_summary=summary,
                scope='Saved identities, independent confusion counts, matched exposure, selection, exact selected development replay, persistent BN counters; no population significance claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--test',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.run,a.data_dir,a.test)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],complete=result['complete'],checks=len(result['checks']))))
