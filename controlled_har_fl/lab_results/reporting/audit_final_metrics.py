"""Recompute all reported held-out metrics from immutable saved predictions, independently of training code."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'lab_results'
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metrics(y,guess):
 matrix=np.bincount(y*15+guess,minlength=225).reshape(15,15)
 support=matrix.sum(1);predicted=matrix.sum(0);tp=matrix.diagonal()
 recall=np.divide(tp,support,out=np.zeros(15),where=support>0)
 precision=np.divide(tp,predicted,out=np.zeros(15),where=predicted>0)
 f1=np.divide(2*tp,support+predicted,out=np.zeros(15),where=(support+predicted)>0)
 return dict(n=int(matrix.sum()),accuracy=float(tp.sum()/matrix.sum()),macro_f1_all_declared_classes=float(f1.mean()),
  per_class_support=support.tolist(),per_class_recall=recall.tolist(),per_class_precision=precision.tolist(),per_class_f1=f1.tolist(),
  fall_recall=float(recall[0]) if support[0] else None,fall_precision=float(precision[0]) if predicted[0] else None,confusion_matrix=matrix.tolist())
def main():
 audit=json.loads((OUT/'final_audit.json').read_text());assert audit['passed'] and audit['complete']
 frozen=json.loads((OUT/'frozen_selection.json').read_text());expected={(r['seed'],r['method']):r for r in frozen['checkpoints']}
 report=json.loads((OUT/'final_test/results.json').read_text());assert report['complete']
 assert report['frozen_selection_sha256']==sha(OUT/'frozen_selection.json')
 contract=frozen['run_contract'];assert len(report['results'])==len(contract['methods'])*len(contract['seeds'])
 with np.load(OUT/'final_test/allocation.npz',allow_pickle=False) as a:
  identity={str(s):(int(y),int(p),int(c)) for s,y,p,c in zip(a['sequence_ids'],a['y'],a['subject_ids'],a['client_ids'])}
 rows=[];inputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'frozen_selection.json',OUT/'final_audit.json',OUT/'final_test/results.json',OUT/'final_test/allocation.npz']}
 seen=set()
 for result in report['results']:
  key=(result['seed'],result['method']);assert key not in seen;seen.add(key);selection=expected[key]
  for name in ('selected_round','checkpoint','checkpoint_sha256','client_state_sha256','server_state_sha256'):assert result[name]==selection[name]
  assert sha(ROOT/selection['checkpoint'])==selection['checkpoint_sha256']
  wanted=[selection['server_state_sha256']] if result['method'] in ('fedavg','fedprox') else selection['client_state_sha256']
  assert result['state_hashes_before']==result['state_hashes_after']==wanted
  path=OUT/f"final_test/seed_{result['seed']}_{result['method']}.npz"
  assert sha(path)==result['prediction_sha256'];inputs[str(path.relative_to(ROOT))]=sha(path)
  with np.load(path,allow_pickle=False) as a:p=dict(a)
  ids=p['sequence_ids'].tolist();assert len(ids)==len(set(ids))==len(identity) and set(ids)==set(identity)
  assert all(identity[s]==(int(y),int(subject),int(cid)) for s,y,subject,cid in zip(ids,p['y'],p['subject_ids'],p['client_ids']))
  assert p['logits'].shape==(len(ids),15) and np.isfinite(p['logits']).all()
  guess=p['logits'].argmax(1);actual=metrics(p['y'],guess);saved=result['metrics']
  for name,value in actual.items():assert saved['pooled'][name]==value,(key,name)
  scores=p['logits'].astype(np.float64);maximum=scores.max(1)
  ce=float(np.mean(maximum+np.log(np.exp(scores-maximum[:,None]).sum(1))-scores[np.arange(len(ids)),p['y']]))
  assert abs(ce-saved['pooled']['cross_entropy'])<=1e-5
  clients=[]
  for cid in range(contract['protocol']['num_clients']):
   mask=p['client_ids']==cid;cm=metrics(p['y'][mask],guess[mask]);assert cm==saved['clients'][cid];clients.append(cm)
  assert float(np.mean([m['accuracy'] for m in clients]))==saved['mean_client_accuracy']
  assert min(m['accuracy'] for m in clients)==saved['worst_client_accuracy']
  assert float(np.mean([m['macro_f1_all_declared_classes'] for m in clients]))==saved['mean_client_macro_f1']
  assert min(m['macro_f1_all_declared_classes'] for m in clients)==saved['worst_client_macro_f1']
  subjects=[]
  for subject in np.unique(p['subject_ids']):
   mask=p['subject_ids']==subject;subjects.append(dict(subject_id=int(subject),n=int(mask.sum()),accuracy=float((guess[mask]==p['y'][mask]).mean())))
  assert subjects==saved['subjects'];assert float(np.mean([s['accuracy'] for s in subjects]))==saved['mean_subject_accuracy']
  matrix=np.asarray(actual['confusion_matrix']);fall=dict(tp=int(matrix[0,0]),fn=int(matrix[0].sum()-matrix[0,0]),fp=int(matrix[:,0].sum()-matrix[0,0]),support=int(matrix[0].sum()))
  assert fall==saved['falling_counts']
  rows.append(dict(seed=key[0],method=key[1],pooled=actual,clients=clients,subjects=subjects,falling_counts=fall,cross_entropy_float64=ce))
 assert seen==set(expected)
 result=dict(passed=True,scope='Independent NumPy reconstruction of every pooled/per-class/client/subject/falling metric; prediction identities and selected model hashes checked. No training-code metric functions imported.',cross_entropy_absolute_tolerance=1e-5,all_other_reported_metrics_exact=True,inputs=inputs,source_sha256=sha(__file__),results=rows)
 (OUT/'reporting/final_metric_audit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'passed':True,'method_seed_results':len(rows)}))
if __name__=='__main__':main()
