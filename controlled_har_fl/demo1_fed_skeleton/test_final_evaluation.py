"""Adversarial checks of the development/final-test evidence boundary."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import corrected_fl as fl, development_fl as dev, final_evaluation as final
from .test_corrected_fl import toy_model, toy_shard
from .prepare_corrected_fl import class_contract


class FinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def inputs(self):
        return ([toy_shard(prefix='train0'),toy_shard(3.,prefix='train1')],
                [toy_shard(.5,prefix='dev0',subject_offset=20),toy_shard(3.5,prefix='dev1',subject_offset=20)])

    def test_rejects_identity_overlap_and_duplicate(self):
        train,development=self.inputs()
        for ids,subjects in [(train[0].sequence_ids,np.full(4,90)),
                             (np.array(['a','b','c','d']),development[0].subject_ids),
                             (np.array(['a','a']),np.array([90,91]))]:
            with self.assertRaises(ValueError):final.validate_identities(ids,subjects,train,development)

    def test_wrong_class_map_rejected(self):
        mapping,names=class_contract(15)
        protocol={'label_names':names,'ntu_zero_based_ids':list(mapping)}
        final.validate_protocol(protocol)
        protocol['label_names']=list(reversed(names))
        with self.assertRaisesRegex(ValueError,'class map'):final.validate_protocol(protocol)

    def test_allocation_is_fixed_and_invalid_clients_rejected(self):
        labels=np.tile(np.arange(2,dtype=np.int64),20)
        a=final.allocate(labels,[[.8,.2],[.3,.7]],20261024)
        np.testing.assert_array_equal(a,final.allocate(labels,[[.8,.2],[.3,.7]],20261024))
        self.assertEqual(np.bincount(a[labels==0]).tolist(),[16,4])
        for bad in (np.array([-1,0]),np.array([0,2]),np.array([0.,1.]),np.array([0,0])):
            with self.assertRaisesRegex(ValueError,'assignment'):final.validate_clients(bad,2)
        with self.assertRaises(ValueError):final.allocate(labels,[[1.,1.],[.3,.7]],1)

    def test_fixed_inference_matches_direct_counts_and_cannot_select(self):
        train,development=self.inputs()
        _,artifacts=dev.run_method(toy_model,train,development,method='fedbn',rounds=2,
            local_epochs=1,batch_size=2,lr=.01,seed=42,device='cpu',num_classes=2)
        checkpoint=artifacts['best_development'][1]
        original=copy.deepcopy(checkpoint)
        models=[toy_model() for _ in train]
        for m,state in zip(models,checkpoint['clients']):m.load_state_dict(state)
        metrics,predictions,before,after=final.score_fixed(models,development,'fedbn',device='cpu',batch_size=2,num_classes=2)
        matrix=np.zeros((2,2),dtype=np.int64)
        for truth,pred in zip(predictions['y'],predictions['logits'].argmax(1)):matrix[truth,pred]+=1
        self.assertEqual(metrics['pooled']['confusion_matrix'],matrix.tolist())
        self.assertEqual(metrics['pooled']['accuracy'],np.trace(matrix)/matrix.sum())
        f1=np.divide(2*np.diag(matrix),matrix.sum(0)+matrix.sum(1),out=np.zeros(2),where=(matrix.sum(0)+matrix.sum(1))>0)
        self.assertAlmostEqual(metrics['pooled']['macro_f1_all_declared_classes'],float(f1.mean()))
        self.assertEqual(before,after)
        for s in development:s.y[:]=0
        final.score_fixed(models,development,'fedbn',device='cpu',batch_size=2,num_classes=2)
        self.assertEqual(checkpoint['round'],original['round'])
        self.assertEqual([dev.state_hash(s) for s in checkpoint['clients']],[dev.state_hash(s) for s in original['clients']])

    def test_changed_checkpoint_and_bn_state_rejected(self):
        train,development=self.inputs()
        _,artifacts=dev.run_method(toy_model,train,development,method='fedbn',rounds=1,
            local_epochs=1,batch_size=2,lr=.01,seed=42,device='cpu',num_classes=2)
        state=artifacts['best_development'][1]
        with tempfile.TemporaryDirectory() as tmp,patch.object(final,'ROOT',Path(tmp)):
            path=Path(tmp)/'selected.pt'; torch.save(state,path)
            record=dict(checkpoint=path.name,checkpoint_sha256=fl.sha256(path),client_state_sha256=[dev.state_hash(s) for s in state['clients']],
                        server_state_sha256=dev.state_hash(state['server']),selected_round=1,seed=42,method='fedbn')
            final.load_selected(record,toy_model,2)
            with path.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint hash'):final.load_selected(record,toy_model,2)
            state['clients'][0]['0.running_mean']+=1
            torch.save(state,path); record['checkpoint_sha256']=fl.sha256(path)
            with self.assertRaisesRegex(ValueError,'BN state'):final.load_selected(record,toy_model,2)

    def test_inference_bn_mutation_rejected(self):
        _,heldout=self.inputs()
        models=[toy_model(),toy_model()]
        evaluate=dev.evaluate
        def mutate(*args,**kwargs):
            result=evaluate(*args,**kwargs)
            models[0][0].running_mean.add_(1)
            return result
        with patch.object(dev,'evaluate',side_effect=mutate),self.assertRaisesRegex(ValueError,'altered'):
            final.score_fixed(models,heldout,'local',device='cpu',batch_size=2,num_classes=2)

    def test_development_ties_use_lower_loss_then_earliest(self):
        train,heldout=self.inputs()
        evaluate=dev.evaluate
        for losses,wanted in [([1.,1.,1.],1),([2.,1.,1.],2)]:
            iterator=iter(losses)
            def tied(*args,**kwargs):
                metrics,predictions=evaluate(*args,**kwargs)
                metrics['pooled']['macro_f1_all_declared_classes']=.5
                metrics['pooled']['cross_entropy']=next(iterator)
                return metrics,predictions
            with patch.object(dev,'evaluate',side_effect=tied):
                result,_=dev.run_method(toy_model,train,heldout,method='fedbn',rounds=3,
                    local_epochs=1,batch_size=2,lr=.01,seed=42,device='cpu',num_classes=2)
            self.assertEqual(result['selected_round'],wanted)


if __name__=='__main__':unittest.main()
