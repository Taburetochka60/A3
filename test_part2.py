import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

import part2
import part2_data as data
from ner_data import IGNORE, collate_ner, decode_bio, tokenize_offsets
from pilot_ner import documents
from recurrent_model import NERModel, RecurrentEncoder


class Part2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fixed_configuration(self):
        c = part2.CONFIG
        self.assertEqual(c['encoder'], dict(rnn_type='lstm', num_layers=2, embedding_dim=64,
                                           hidden_size=64, dropout=.2, bidirectional=True))
        self.assertEqual((c['optimizer'], c['lr'], c['weight_decay']), ('AdamW', .002, .01))
        self.assertEqual((c['train_rows'], c['batch_size'], c['epochs'], c['patience'], c['seed']), (100000,128,15,6,42))
        self.assertEqual((c['threshold'], c['prediction_filter']), (.8,'strict'))
        changed = part2.configuration(num_layers=1, lr=.0005)
        self.assertEqual(changed['encoder']['num_layers'],1)
        self.assertEqual(c['encoder']['num_layers'],2)

    def test_manifest_and_grouping_guards(self):
        path = Path(__file__).with_name('part2_split.json')
        spec = data.read_split(path)
        self.assertEqual(len(spec['fit']),100000)
        self.assertEqual(set(spec['cv_folds']),{0,1,2})
        with tempfile.TemporaryDirectory() as folder:
            changed = copy.deepcopy(spec); changed['cv_folds'][0]=(changed['cv_folds'][0]+1)%3
            target=Path(folder)/'split.json';data.save_json(target,changed)
            with self.assertRaisesRegex(ValueError,'content'):
                data.read_split(target)
        with patch('part2_data.grouping_hash',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'Grouping'):
                data.read_split(path)
        self.assertEqual(data.text_group('М.Київ'),data.text_group('місто Київ'))
        self.assertNotEqual(data.text_group('Миру 12/14'),data.text_group('Миру 12-14'))

    def test_word_and_bpe_fit_only_training_and_keep_offsets(self):
        train = documents(pd.DataFrame(dict(text=['Київ Львів']*5,loc_markers=['[(0,4)]']*5)))
        heldout = documents(pd.DataFrame(dict(text=['Ω'],loc_markers=['[]'])))
        for kind in ['word','word_raw','bpe']:
            state=data.fit_encoding(train,kind)
            self.assertEqual(data.encode_documents(heldout,state)[0]['ids'],[1])
            text='😀 Киі\u0308в, Миру 12\nЛьвів'
            docs=documents(pd.DataFrame(dict(text=[text],loc_markers=['[]'])))
            encoded=data.encode_documents(docs,state)[0]
            self.assertTrue(all(text[t.start:t.end]==t.value for t in encoded['tokens']))
            self.assertTrue(all(any(t.start<=i<t.end for t in encoded['tokens']) for i,c in enumerate(text) if not c.isspace()))
            self.assertEqual(decode_bio(encoded['tokens'],encoded['bio'])[0],[])
        before=data.documents_hash([dict(train[0],group='g')])
        changed=dict(train[0],group='g',spans=[(5,10)])
        self.assertNotEqual(before,data.documents_hash([changed]))

    def test_shapes_masking_padding_and_gradients(self):
        for kind in ['gru','lstm']:
            for layers in [1,2]:
                model=NERModel(20,rnn_type=kind,num_layers=layers,embedding_dim=8,hidden_size=6).eval()
                ids,lengths,targets=collate_ner([([2,3],[1,2]),([4,5,6],[0,1,2])])
                logits=model(ids,lengths)
                self.assertEqual(logits.shape,(2,3,3))
                self.assertEqual(model.encoder.output_dim,12)
                torch.testing.assert_close(logits[0,:2],model(torch.tensor([[2,3,0,0]]),torch.tensor([2]))[0,:2])
                loss=nn.functional.cross_entropy(logits.reshape(-1,3),targets.reshape(-1))
                torch.testing.assert_close(loss,nn.functional.cross_entropy(logits[targets!=IGNORE],targets[targets!=IGNORE]))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertEqual(model.encoder.embedding.weight.grad[0].abs().sum(),0)

    def test_original_gru_initial_weights(self):
        torch.manual_seed(42)
        embedding=nn.Embedding(20,8,padding_idx=0);rnn=nn.GRU(8,6,batch_first=True,bidirectional=True)
        torch.manual_seed(42)
        model=RecurrentEncoder(20,embedding_dim=8,hidden_size=6)
        self.assertTrue(torch.equal(model.embedding.weight,embedding.weight))
        for name,value in rnn.state_dict().items():
            self.assertTrue(torch.equal(model.rnn.state_dict()[name],value))

    def test_buckets_and_rng_resume(self):
        examples=[([1]*(i%7+1),[0]*(i%7+1)) for i in range(43)]
        batches=part2.batches(examples,8,42,1)
        self.assertEqual(sorted(i for b in batches for i in b),list(range(43)))
        self.assertEqual(batches,part2.batches(examples,8,42,1))
        torch.manual_seed(42);np.random.seed(42)
        state=part2.rng_state();first=(torch.rand(4),np.random.rand(4))
        part2.restore_rng(state)
        self.assertTrue(torch.equal(first[0],torch.rand(4)))
        np.testing.assert_array_equal(first[1],np.random.rand(4))

    def test_finite_update_and_optimizer_checkpoint(self):
        torch.manual_seed(42)
        model=NERModel(10,rnn_type='lstm',num_layers=2,embedding_dim=4,hidden_size=4)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        batch=collate_ner([([2,3],[1,2]),([4],[0])])
        result=part2.update(model,batch,optimizer,scaler,torch.ones(3),'fp32')
        self.assertTrue(np.isfinite(result).all())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),path)
            saved=torch.load(path,weights_only=True)
            other=NERModel(10,rnn_type='lstm',num_layers=2,embedding_dim=4,hidden_size=4)
            other.load_state_dict(saved['model'])
            for a,b in zip(model.parameters(),other.parameters()):self.assertTrue(torch.equal(a,b))

    def test_cpu_training_and_protected_validation_are_rejected(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Colab CUDA'):part2.gpu()
        tr=[dict(row_id=1,group='fit')];va=[dict(row_id=9,group='audit')]
        spec=dict(content_sha256='fixture',fit=[1],checkpoint=[2])
        with patch('part2.gpu',return_value={}):
            with self.assertRaisesRegex(ValueError,'Epoch selection'):
                part2.train(part2.CONFIG,tr,va,spec,'unused')

    def test_decoding_filters_threshold_and_custom_decoder(self):
        text='Івано-Франківської області 😀\n12'
        tokens=tokenize_offsets(text);doc=dict(text=text,tokens=tokens,spans=[])
        p=np.eye(3)[[1]*len(tokens)]
        original=part2.decode([doc],[p])[0][0]
        strict=part2.decode([doc],[p],.8,'strict')[0][0]
        context=part2.decode([doc],[p],.8,'context')[0][0]
        self.assertIn('-',original);self.assertNotIn('-',strict);self.assertIn('-',context)
        self.assertNotIn('😀',strict);self.assertNotIn('\n',strict);self.assertIn('12',strict)
        values,spans,_=part2.decode([doc],[p],.8,'strict')
        self.assertTrue(all(text[a:b]==v for (a,b),v in zip(spans[0],values[0])))
        self.assertEqual(part2.decode([doc],[p],decoder=lambda tokens,tags:([],0))[0],[[]])
        self.assertEqual(part2.decode([doc],[p],1.01,'strict')[0],[[]])

    def test_submission_reference_hash_prevents_overwrite(self):
        docs=[dict(text='Київ',tokens=tokenize_offsets('Київ'),text_id=i) for i in range(477)]
        p=[np.array([[.1,.9,0.]]) for _ in docs]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'submission.csv'
            part2.export_submission(docs,p,path)
            baseline=path.read_bytes();digest=data.file_hash(path)
            part2.export_submission(docs,p,path,expected_hash=digest)
            with self.assertRaisesRegex(ValueError,'reference'):
                part2.export_submission(docs,p,path,expected_hash='wrong')
            self.assertEqual(path.read_bytes(),baseline)
            self.assertEqual(pd.read_csv(path).text_id.nunique(),477)
            self.assertTrue(all(json.loads(v)==['Київ'] for v in pd.read_csv(path).locations))


if __name__=='__main__':
    unittest.main()
