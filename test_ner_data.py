import unittest
import numpy as np
import torch

from ner_data import (tokenize_offsets,encode_bio,decode_bio,window_ranges,
                      training_windows,collate_ner,merge_window_probabilities,IGNORE)
from recurrent_model import NERModel


class NERDataTests(unittest.TestCase):
    def test_unicode_offsets_original_substrings(self):
        text='😀 Киі\u0308в, вул._Миру 12\nЛьвів'
        tokens=tokenize_offsets(text)
        self.assertTrue(all(text[t.start:t.end]==t.value for t in tokens))
        self.assertTrue(all(any(t.start<=i<t.end for t in tokens) for i,c in enumerate(text) if not c.isspace()))
        a=text.index('К');b=text.index(',')
        labels,trimmed,_,issues=encode_bio(text,[(a,b)])
        self.assertFalse(issues)
        self.assertEqual(decode_bio(tokens,labels)[0],[(a,b)])

    def test_adjacent_entities_and_whitespace(self):
        text=' Київ Львів '
        labels,trimmed,changes,issues=encode_bio(text,[(0,5),(6,12)])
        self.assertFalse(issues);self.assertEqual(labels.tolist(),[1,1])
        self.assertEqual(trimmed,[(1,5),(6,11)])
        self.assertEqual(len(changes),2)

    def test_partial_word_is_not_silently_repaired(self):
        text='Нью-Йорка'
        labels,_,_,issues=encode_bio(text,[(0,8)])
        self.assertIsNone(labels);self.assertIn('partial_token_boundary',issues)
        self.assertEqual([t.value for t in tokenize_offsets(text)],['Нью','-','Йорка'])

    def test_long_windows_supervise_every_token_once(self):
        for n in [0,1,127,128,129,256,1001]:
            windows=list(window_ranges(n));counts=np.zeros(n,dtype=int)
            for (a,b),(_,labels) in zip(windows,training_windows(list(range(n)),[0]*n)):
                self.assertLessEqual(b-a,128)
                counts[a:b]+=np.array(labels)!=IGNORE
            np.testing.assert_array_equal(counts,np.ones(n,dtype=int))

    def test_entity_crossing_window_is_recovered_globally(self):
        text=' '.join(['текст']*260)
        tokens=tokenize_offsets(text);tags=np.zeros(260,dtype=int);tags[125]=1;tags[126:136]=2
        windows=list(window_ranges(len(tokens)))
        probs=[np.eye(3)[tags[a:b]] for a,b in windows]
        merged=merge_window_probabilities(len(tokens),windows,probs)
        spans,_=decode_bio(tokens,merged.argmax(1))
        self.assertEqual(spans,[(tokens[125].start,tokens[135].end)])
        self.assertTrue(merged[-1,0]==1)

    def test_ner_padding_logits_loss_and_gradients(self):
        torch.manual_seed(42)
        model=NERModel(20,embedding_dim=8,hidden_size=6,dropout=0).eval()
        ids,lengths,targets=collate_ner([([2,3],[1,2]),([4,5,6,7],[0,0,1,2])])
        logits=model(ids,lengths)
        single=model(torch.tensor([[2,3]]),torch.tensor([2]))
        self.assertTrue(torch.allclose(single[0],logits[0,:2],atol=1e-6))
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,3),targets.reshape(-1),ignore_index=IGNORE)
        mask=targets!=IGNORE
        reference=torch.nn.functional.cross_entropy(logits[mask],targets[mask])
        self.assertTrue(torch.allclose(loss,reference))
        model.zero_grad(set_to_none=True);loss.backward()
        self.assertTrue(torch.equal(model.encoder.embedding.weight.grad[0],torch.zeros(8)))
        changed=logits.detach().clone();changed[~mask]=torch.tensor([1000.,-1000.,500.])
        self.assertTrue(torch.allclose(torch.nn.functional.cross_entropy(changed.reshape(-1,3),targets.reshape(-1),ignore_index=IGNORE),loss.detach()))


if __name__=='__main__':unittest.main()
