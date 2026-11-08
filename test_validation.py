"""Checks that matter for held-out recurrent inference and NER uncertainty."""
import unittest
import numpy as np
import torch

from recurrent_model import DomainModel,NERModel
from evaluate_validation import counts_per_text,f1_counts,group_bootstrap
from metrics import comp_metric


class ValidationTests(unittest.TestCase):
    def test_padded_inference_is_invariant(self):
        torch.manual_seed(42)
        model=DomainModel(20,embedding_dim=8,hidden_size=6,dropout=.2).eval()
        short=torch.tensor([[2,3,4]])
        padded=torch.tensor([[2,3,4,0,0]])
        with torch.no_grad():
            a=model(short,torch.tensor([3]));b=model(padded,torch.tensor([3]))
        self.assertTrue(torch.allclose(a,b,atol=1e-6))

    def test_ner_and_domain_share_encoder(self):
        ner=NERModel(20,embedding_dim=8,hidden_size=6)
        domain=DomainModel(20,embedding_dim=8,hidden_size=6)
        self.assertEqual([(n,p.shape) for n,p in ner.encoder.named_parameters()],[(n,p.shape) for n,p in domain.encoder.named_parameters()])
        self.assertEqual(tuple(ner(torch.tensor([[2,3,0],[4,5,6]]),torch.tensor([2,3])).shape),(2,3,3))

    def test_count_aggregation_matches_published_metric(self):
        gold=[['Київ','Львів'],[],['Київ','Київ']]
        pred=[['Київ'],['Одеса'],['київ']]
        self.assertAlmostEqual(float(f1_counts(counts_per_text(gold,pred).sum(0))),comp_metric(gold,pred),places=7)

    def test_group_bootstrap_keeps_duplicate_rows_together(self):
        gold=[['Київ'],['Київ'],['Львів']]
        pred=[['Київ'],['Київ'],['Львів']]
        result=group_bootstrap(gold,pred,['same','same','other'],repetitions=100)
        self.assertEqual(result['bootstrap_groups'],2)
        np.testing.assert_equal(result['f1_95ci'],[1,1])


if __name__=='__main__':unittest.main()
