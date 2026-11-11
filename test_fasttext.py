import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import part2_data as source
from ner_data import collate_ner, token_key
from pilot_ner import documents, vocabulary
from experiments import fasttext_embeddings as embeddings
from experiments import fasttext_experiment as experiment


class Provider:
    def get_word_vector(self,word):
        result=np.zeros(300,dtype=np.float32);result[sum(map(ord,word))%300]=2.
        return result
    def get_word_id(self,word):return 2 if word=='київ' else -1


class FastTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self,count=3):
        fit=documents(pd.DataFrame(dict(text=['Київ Львів']*count,loc_markers=['[(0,4)]']*count)))
        valid=documents(pd.DataFrame(dict(text=['Ω Київ'],loc_markers=['[(2,6)]']),index=[900]))
        for d in fit:d['group']='g'+str(d['row_id'])
        valid[0]['group']='protected'
        return dict(fit=fit,checkpoint=valid),vocabulary(fit)

    def test_matching_initialization_and_padding(self):
        docs,words=self.fixture();a,b,info=embeddings.matrices(words,Provider())
        self.assertTrue(np.array_equal(a[1],b[1]));self.assertEqual(a.shape,b.shape)
        np.testing.assert_allclose(np.linalg.norm(a[1:],axis=1),1,atol=1e-6)
        m1=experiment.build_model(a);m2=experiment.build_model(b)
        self.assertEqual(experiment.recurrent_hash(m1),experiment.recurrent_hash(m2))
        self.assertFalse(torch.equal(m1.encoder.embedding.weight[2:],m2.encoder.embedding.weight[2:]))
        self.assertEqual(int(torch.count_nonzero(m2.encoder.embedding.weight[0])),0)
        self.assertEqual(info['dimensions'],300);self.assertTrue(info['trainable'])

    def test_pretrained_dimensions_values_and_no_fallback(self):
        with patch.object(Provider,'get_word_vector',return_value=np.zeros(64)):
            with self.assertRaisesRegex(ValueError,'Invalid pretrained'):embeddings.matrices({'київ':2},Provider())
        with patch.object(Provider,'get_word_vector',side_effect=RuntimeError('loading failure')):
            with self.assertRaisesRegex(RuntimeError,'loading failure'):embeddings.matrices({'київ':2},Provider())
        with patch.object(Provider,'get_word_vector',return_value=np.zeros(300)):
            _,matrix,info=embeddings.matrices({'київ':2},Provider());self.assertEqual(matrix[2].sum(),0);self.assertEqual(info['zero_vectors'],1)

    def test_offsets_bio_and_train_only_vocabulary(self):
        docs,words=self.fixture()
        self.assertNotIn('ω',words)
        encoded=source.encode_documents(docs['checkpoint'],dict(kind='word',vocabulary=words))
        self.assertEqual(encoded[0]['ids'][0],1)
        self.assertIs(encoded[0]['tokens'],docs['checkpoint'][0]['tokens'])
        self.assertIs(encoded[0]['bio'],docs['checkpoint'][0]['bio'])
        text='Івано-Франківської області 😀\n12';d=documents(pd.DataFrame(dict(text=[text],loc_markers=['[(0,25)]'])))[0]
        self.assertTrue(all(text[t.start:t.end]==t.value for t in d['tokens']))
        self.assertEqual(experiment.CONFIG['threshold'],.8);self.assertEqual(experiment.CONFIG['prediction_filter'],'strict')

    def test_real_optimizer_step_gradient_and_checkpoint(self):
        docs,words=self.fixture();matrix,_,_=embeddings.matrices(words,Provider());model=experiment.build_model(matrix)
        enc=source.encode_documents(docs['fit'],dict(kind='word',vocabulary=words));batch=collate_ner([(d['ids'],d['bio']) for d in enc])
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        before=model.encoder.rnn.weight_ih_l0.detach().clone()
        values=experiment.base.update(model,batch,optimizer,torch.amp.GradScaler('cuda',enabled=False),torch.ones(3),'fp32')
        self.assertTrue(np.isfinite(values).all());self.assertFalse(torch.equal(before,model.encoder.rnn.weight_ih_l0))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(model.encoder.embedding.weight.grad[0].abs().sum(),0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt';experiment.save_new_torch(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict()),tmp)
            saved=torch.load(path,weights_only=True);other=experiment.build_model(matrix);other.load_state_dict(saved['model'])
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(model.parameters(),other.parameters())))
            with self.assertRaisesRegex(ValueError,'already exists'):experiment.save_new_torch(path,saved,tmp)

    def test_metric_oov_surface_sets_and_bootstrap(self):
        docs,words=self.fixture();ds=docs['checkpoint'];p=[np.eye(3)[d['bio']] for d in ds]
        metrics=experiment.summarize(ds,p,words);self.assertEqual(metrics['f1'],1);self.assertEqual(metrics['iv_recall'],1)
        counts=experiment.metric_counts(ds,p,words)
        interval=experiment.paired_bootstrap(counts,counts,[d['group'] for d in ds],20)
        self.assertEqual((interval['low'],interval['high']),(0,0))
        with self.assertRaisesRegex(ValueError,'Unpaired'):experiment.paired_bootstrap(counts,counts,[],20)

    def test_incompatible_protected_row_is_not_filtered(self):
        doc=documents(pd.DataFrame(dict(text=['Київ'],loc_markers=['[(0,1)]'])))[0]
        self.assertIsNone(doc['bio'])
        metric=experiment.summarize([doc],[np.array([[1.,0.,0.]])],{'київ':2})
        self.assertEqual(metric['rows'],1);self.assertEqual(metric['bio_incompatible'],1)
        self.assertEqual(metric['fn'],1);self.assertIsNone(metric['val_loss'])

    def test_integrity_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'meta.json';embeddings.immutable_json(p,dict(a=1));embeddings.immutable_json(p,dict(a=1))
            with self.assertRaisesRegex(ValueError,'differs'):embeddings.immutable_json(p,dict(a=2))
            p=Path(tmp)/'bad.bin';p.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'checksum'):embeddings.load_fasttext(p,'wrong')
        source.read_split('part2_split.json')

    def test_cpu_guard(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Colab CUDA'):experiment.run_training({},None,None,None,None,None,None,None)

    def test_mid_epoch_resume_and_completed_reuse(self):
        docs,words=self.fixture(5);matrix,_,_=embeddings.matrices(words,Provider())
        config=copy.deepcopy(experiment.CONFIG);config.update(batch_size=2,epochs=2,patience=6)
        actual_check=experiment.check_time;calls=[0]
        def interrupt(deadline):
            calls[0]+=1
            if calls[0]==2:raise experiment.BudgetEnd('fixture interruption')
        class NoTracking:
            def __init__(self,*args,**kwargs):pass
            def log(self,*args):pass
            def finish(self,*args):pass
        with tempfile.TemporaryDirectory() as tmp,patch.object(experiment,'Tracker',NoTracking):
            root=Path(tmp)
            with patch.object(experiment,'check_time',interrupt):
                with self.assertRaises(experiment.BudgetEnd):
                    experiment.fit_arm('random',docs,words,matrix,root/'resumed','fixture',{},root,'cpu',time.monotonic()+300,config)
            pointer=json.loads((root/'resumed/resume.json').read_text())
            saved=torch.load(root/'resumed'/pointer['file'],weights_only=True)
            self.assertEqual(saved['progress']['batch'],1)
            self.assertFalse((root/'resumed/result.json').exists())
            resumed=experiment.fit_arm('random',docs,words,matrix,root/'resumed','fixture',{},root,'cpu',time.monotonic()+300,config)
            full=experiment.fit_arm('random',docs,words,matrix,root/'full','fixture',{},root,'cpu',time.monotonic()+300,config)
            a=torch.load(root/'resumed'/resumed['checkpoint'],weights_only=True)['model'];b=torch.load(root/'full'/full['checkpoint'],weights_only=True)['model']
            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
            self.assertEqual(resumed['best_epoch'],full['best_epoch'])
            with patch.object(experiment,'build_model',side_effect=AssertionError('Completed run retrained')):
                self.assertEqual(experiment.fit_arm('random',docs,words,matrix,root/'resumed','fixture',{},root,'cpu',time.monotonic()+300,config),resumed)

    @unittest.skipUnless(importlib.util.find_spec('fasttext'),'Install requirements-fasttext.txt for real binary smoke test')
    def test_real_fasttext_binary_and_subword_oov(self):
        import fasttext
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'corpus.txt';p.write_text(('київ львів україна київський львівський\n')*100)
            ft=fasttext.train_unsupervised(str(p),model='cbow',dim=300,minCount=1,epoch=0,bucket=1000,minn=5,maxn=5,thread=1,verbose=0,t=1.,lr=.001)
            binary=Path(tmp)/'tiny.bin';ft.save_model(str(binary))
            loaded,verified=embeddings.load_fasttext(binary,source.file_hash(binary),official=False)
            self.assertEqual(loaded.get_dimension(),300);self.assertEqual(verified['header']['minn'],5)
            unknown='київськомуневідомому'
            self.assertEqual(loaded.get_word_id(unknown),-1)
            self.assertGreater(np.linalg.norm(embeddings.checked_vector(loaded,unknown)),0)
            with self.assertRaisesRegex(ValueError,'expected official'):embeddings.load_fasttext(binary,source.file_hash(binary))

    def test_notebook_and_source_manifest(self):
        path=Path('A3_fasttext_colab.ipynb')
        manifest=Path('fasttext_current_source_manifest.json') if path.exists() else Path('source_manifest.json')
        experiment.verify_source('.',manifest)
        if not path.exists():return
        nb=json.loads(path.read_text());codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        self.assertLess(max(map(len,codes)),4000)
        for code in codes:
            compile(code,str(path),'exec');self.assertNotIn('base64',code);self.assertNotIn('BUNDLE',code)
        self.assertIn(source.file_hash(manifest),codes[0])
        self.assertIn('fasttext_current_source',codes[0])
        self.assertIn('results/fasttext_reproduction',codes[2])
        historical=json.loads(Path('fasttext_source_manifest.json').read_text())
        self.assertEqual(historical['source_id'],source.fingerprint(historical['files']))
        self.assertNotEqual(historical['source_id'],json.loads(manifest.read_text())['source_id'])
        for name in ['part2.py','part2_split.json','ner_data.py','pilot_ner.py','recurrent_model.py','metrics.py','validation.py']:
            self.assertEqual(source.file_hash(name),historical['files'][name])

    def test_current_source_copy_checks_every_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=experiment.make_source('.',tmp)
            self.assertEqual(experiment.verify_source(tmp,Path(tmp)/'source_manifest.json'),manifest['source_id'])
            (Path(tmp)/'metrics.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Source hash mismatch'):
                experiment.verify_source(tmp,Path(tmp)/'source_manifest.json')


if __name__=='__main__':unittest.main()
