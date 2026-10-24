"""Agent-managed primary run, evidence checks, frozen final evaluation, and local export."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'lab_results'
PY=ROOT/'.venv-har/bin/python'


def main():
    p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,required=True)
    p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);p.add_argument('--deadline',required=True)
    a=p.parse_args()
    data='experiments/har_fl_repair_20261001/full_development_data'
    run='lab_results/primary'
    commands=[]
    def stage(name,args):
        cmd=[str(PY),*args]
        commands.append(dict(stage=name,command=cmd,started_utc=datetime.now(timezone.utc).isoformat()))
        (OUT/'pipeline_commands.json').write_text(json.dumps(commands,indent=2))
        (OUT/'STATUS.md').write_text(f'# HAR/FL lab status\n\nStage: {name}.\n\n'
            'Training progress: primary/status.json and primary.log. Original bundle is preserved.\n'
            'Primary protocol: primary/contract.json. No test-based selection.\n'
            'Setup, pilot, timing and verification reports are retained in this directory.\n')
        with (OUT/f'{name}.log').open('a') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        commands[-1]['completed_utc']=datetime.now(timezone.utc).isoformat()
        (OUT/'pipeline_commands.json').write_text(json.dumps(commands,indent=2))
    error=None
    try:
        stage('primary',['experiments/har_fl_lab_20261024/gpu_queue.py','--data-dir',data,'--output-dir',run,
            '--rounds',str(a.rounds),'--seeds',*[str(s) for s in a.seeds],'--batch-size',str(a.batch_size),'--deadline',a.deadline,'--purpose','primary'])
        status=json.loads((OUT/'primary/status.json').read_text())
        if status['state']!='complete':raise RuntimeError(f'Primary incomplete: {status}')
        stage('development_audit',['experiments/har_fl_lab_20261024/audit_results.py','--run',run,'--data-dir',data,'--output','lab_results/development_audit.json'])
        stage('selection_freeze',['-m','demo1_fed_skeleton.final_evaluation','freeze','--run',run,'--output','lab_results/frozen_selection.json'])
        stage('final_evaluation',['-m','demo1_fed_skeleton.final_evaluation','evaluate','--frozen','lab_results/frozen_selection.json',
            '--data-dir',data,'--output','lab_results/final_test'])
        stage('final_audit',['experiments/har_fl_lab_20261024/audit_results.py','--run',run,'--data-dir',data,
            '--test','lab_results/final_test','--output','lab_results/final_audit.json'])
    except BaseException:
        error=traceback.format_exc()
        (OUT/'pipeline_error.txt').write_text(error)
    brief=['# HAR/FL results brief','',f'Updated UTC: {datetime.now(timezone.utc).isoformat()}','',
           f'Frozen target: {a.rounds} rounds; seeds {a.seeds}; centralized, local, FedAvg, FedBN, FedProx.',
           'Model, graph and preprocessing unchanged. Float32 CUDA, TF32 disabled. No pooled BN calibration or warm start.',
           'Selection uses pooled development macro-F1, then lower cross-entropy, then earliest round.','']
    if error:
        brief+=['Outcome: INCOMPLETE. See pipeline_error.txt and primary/status.json. Do not treat partial development scores as final results.','']
    else:
        audit=json.loads((OUT/'final_audit.json').read_text())
        brief+=['Outcome: completed with independent artifact audit (final_audit.json).','',
                '| Method | Test accuracy mean | Across-seed sample SD | Test macro-F1 mean | Across-seed sample SD |',
                '|---|---:|---:|---:|---:|']
        for method in ('centralized','local','fedavg','fedbn','fedprox'):
            row=audit['test_summary'][method];acc=row['accuracy'];f1=row['macro_f1_all_declared_classes']
            def fmt(x):return f'{x:.6f}' if x is not None else 'not estimable'
            brief.append(f"| {method} | {fmt(acc['mean'])} | {fmt(acc['sample_std'])} | {fmt(f1['mean'])} | {fmt(f1['sample_std'])} |")
        brief+=['','Producing artifacts: final_test/*.npz and *.json; independently reconstructed aggregates and paired seed differences: final_audit.json.',
                'Per-class support/precision/recall, falling counts, client and subject summaries are in each final_test/seed_*_METHOD.json.',
                'Selected and final states, per-round development curves and optimizer-step accounting: primary/seed_*/METHOD/.',
                'The fixed round budget is not a claim of convergence. Across-seed SD is descriptive; no significance or general fairness claim.','']
    brief+=['Limitations: synthetic label-skew cohorts use ground-truth labels for benchmark allocation, not an available deployment router. '
            'The NTU benchmark was historically explored. Custom directed graph retained. '
            'No novel clustered method, physiological validity, synchronized fusion, privacy, deployment throughput or safety claim.',
            '', 'Return archive excludes the original raw/prepared tensor files and local Python environment. Keep the original bundle for reproduction.']
    (OUT/'RESULTS_BRIEF.md').write_text('\n'.join(brief)+'\n')
    (OUT/'STATUS.md').write_text('# HAR/FL lab status\n\n'+('INCOMPLETE; see pipeline_error.txt.' if error else 'Comparison, final evaluation and independent audit completed.')+
        '\n\nSee RESULTS_BRIEF.md for evidence, limitations and completed/missing conditions.\nReturn archive verification: archive_verification.json.\n')
    # This environment was created for this task and is excluded from the return archive.
    # Reclaim it only after all GPU work is finished, if export space needs it.
    output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and p.suffix!='.zip')
    if shutil.disk_usage(OUT).free < output_bytes+500_000_000:
        env=ROOT/'.venv-har'
        if env.is_symlink() or not (OUT/'environment_freeze.txt').is_file():
            raise RuntimeError('Cannot safely reclaim temporary environment')
        shutil.rmtree(env)
        (OUT/'environment_cleanup.json').write_text(json.dumps(dict(removed=str(env),
            reason='Reclaimed task-created environment after GPU verification to fit local export',
            reproduction='environment_freeze.txt and install logs retained; original raw data and source unchanged'),indent=2)+'\n')
    subprocess.run(['/usr/bin/python3','experiments/har_fl_lab_20261024/package_return.py'],cwd=ROOT,check=True)
    if error:sys.exit(1)


if __name__=='__main__':main()
