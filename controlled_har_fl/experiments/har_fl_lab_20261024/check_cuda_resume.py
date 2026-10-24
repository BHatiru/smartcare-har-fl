"""Run the existing exact serialized-resume assertions on CUDA for all five methods."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.har_fl_lab_20261024.gpu_queue import configure, compact_checkpoint
from demo1_fed_skeleton.test_resumable_fl import ResumeTests


class CUDAResumeTests(ResumeTests):
    def run_method(self,method='fedbn',**extra):
        extra['device']='cuda'
        return super().run_method(method,**extra)

    def test_compact_disk_resume_is_exact_for_all_methods(self):
        import torch
        import numpy as np
        from demo1_fed_skeleton import development_fl as dev
        for method in dev.METHODS:
            saved=[]
            expected,endpoints=self.run_method(method,rounds=2)
            self.run_method(method,rounds=1,on_checkpoint=lambda state:saved.append(compact_checkpoint(state)))
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'compact.pt';torch.save(saved[-1],path)
                restored=torch.load(path,map_location='cpu',weights_only=True)
            actual,resumed=self.run_method(method,rounds=2,resume_state=restored)
            self.assertEqual(expected['best_development'],actual['best_development'])
            for endpoint in endpoints:
                np.testing.assert_array_equal(endpoints[endpoint][0]['logits'],resumed[endpoint][0]['logits'])
                self.assertEqual([dev.state_hash(s) for s in endpoints[endpoint][1]['clients']],
                                 [dev.state_hash(s) for s in resumed[endpoint][1]['clients']])


if __name__=='__main__':
    configure()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CUDAResumeTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    Path('lab_results/cuda_resume.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,
        scope='Exact serialized round-boundary CUDA resume; all five methods, toy BN network'),indent=2))
    sys.exit(not result.wasSuccessful())
