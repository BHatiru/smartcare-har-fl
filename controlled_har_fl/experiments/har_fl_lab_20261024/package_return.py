"""Create and read-back verify a local evidence archive; excludes raw tensors and environment."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'lab_results'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    files=[]
    for base in (ROOT/'demo1_fed_skeleton',ROOT/'experiments/har_fl_lab_20261024'):
        files.extend(base.rglob('*.py'))
    files.extend(ROOT/name for name in ('AGENTS.md','START_HERE.md','BUNDLE_MANIFEST.json','handoffs/FL_LAB_5080_20261024.md'))
    repair=ROOT/'experiments/har_fl_repair_20261001'
    files.extend(repair/name for name in ('run_background_fl.py','verify_pilot.py','legacy_test_binding.json','legacy_test_identities.csv'))
    for data in ('pilot_data','full_development_data'):
        files.extend(repair/data/name for name in ('protocol.json','split_manifest.json'))
    files.extend(p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.zip','.tmp')
                 and p.name not in ('RETURN_TO_LAPTOP.sha256','archive_verification.json','CHECKSUMS.json','RUN.lock')
                 and not p.name.endswith('.tmp.npz'))
    files=sorted(set(files))
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    temp=OUT/'RETURN_TO_LAPTOP.tmp'
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for path in files:archive.write(path,str(path.relative_to(ROOT)))
        archive.writestr('CHECKSUMS.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC failure')
        expected=json.loads(archive.read('CHECKSUMS.json'))
        for name,entry in expected.items():
            with archive.open(name) as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=entry['sha256'] or archive.getinfo(name).file_size!=entry['bytes']:
                raise ValueError(f'ZIP content mismatch: {name}')
    target=OUT/'RETURN_TO_LAPTOP.zip'
    temp.replace(target)
    digest=sha(target)
    (OUT/'RETURN_TO_LAPTOP.sha256').write_text(f'{digest}  RETURN_TO_LAPTOP.zip\n')
    report=dict(passed=True,verified_utc=datetime.now(timezone.utc).isoformat(),files=len(files),
                archive_bytes=target.stat().st_size,archive_sha256=digest,
                checks='Every member SHA256 and size read back; ZIP CRC passed',
                excludes='venv, original raw data and prepared tensor files; no credentials')
    (OUT/'archive_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
