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

from ner_data import collate_ner
from pilot_ner import documents
import part2_data as source
from experiments import fasttext_subword as subword
from experiments import fasttext_experiment as experiment
from experiments import fasttext_embeddings as embeddings
from test_fasttext import Provider


class NoTracking:
    def __init__(self,*args,**kwargs):pass
    def log(self,*args):pass
    def finish(self,*args):pass


class SubwordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        fit=documents(pd.DataFrame(dict(text=['Київ рідкіснеслово']*5,loc_markers=['[(0,4)]']*5)))
        valid=documents(pd.DataFrame(dict(text=['Київ новеслово'],loc_markers=['[(5,14)]']),index=[100]))
        for d in fit:d['group']='fit'+str(d['row_id'])
        valid[0]['group']='protected'
        return dict(fit=fit,checkpoint=valid),{'київ':2}

    def representation(self,tmp,provider=None):
        docs,words=self.fixture()
        folder,record=subword.export_vectors(words,docs['fit'],docs['checkpoint'],provider or Provider(),dict(sha256='fixture'),tmp)
        return docs,words,subword.Representation(words,folder,record),folder,record

    def test_train_and_inference_use_same_external_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs,words,rep,folder,record=self.representation(tmp)
            matrix=embeddings.matrices(words,Provider())[1];model=rep.build(matrix)
            both=rep.encode(docs['fit'][:1]+docs['checkpoint'])
            self.assertGreaterEqual(both[0]['ids'][1],len(words)+2)
            self.assertGreaterEqual(both[1]['ids'][1],len(words)+2)
            for doc,encoded in zip(docs['fit'][:1]+docs['checkpoint'],both):
                self.assertIs(doc['tokens'],encoded['tokens']);self.assertIs(doc['bio'],encoded['bio'])
                for token,index in zip(doc['tokens'],encoded['ids']):
                    key=experiment.token_key(token.value)
                    if key not in words:
                        expected=embeddings.unit_rows(embeddings.checked_vector(Provider(),key)[None])[0]
                        np.testing.assert_array_equal(model.encoder.embedding(torch.tensor([index])).detach().numpy()[0],expected)
                    self.assertEqual(doc['text'][token.start:token.end],token.value)
            self.assertEqual(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in experiment.build_model(matrix).parameters()))
            self.assertEqual(experiment.recurrent_hash(model),experiment.recurrent_hash(experiment.build_model(matrix)))
            self.assertNotIn('encoder.embedding.oov',model.state_dict())
            self.assertFalse(model.encoder.embedding.oov.requires_grad)
            with self.assertRaisesRegex(ValueError,'Missing FastText'):
                rep.encode(documents(pd.DataFrame(dict(text=['uncached'],loc_markers=['[]']))))
            future=documents(pd.DataFrame(dict(text=['іншемісто'],loc_markers=['[]'])))
            new_folder,new_record=subword.export_vectors(words,docs['fit'],docs['checkpoint']+future,
                Provider(),dict(sha256='fixture'),tmp)
            expanded=subword.Representation(words,new_folder,new_record);loaded=expanded.build(matrix)
            loaded.load_state_dict(model.state_dict())
            old_p=experiment.infer_probabilities(model,rep.encode(docs['checkpoint']),torch.device('cpu'))
            new_p=experiment.infer_probabilities(loaded,expanded.encode(docs['checkpoint']),torch.device('cpu'))
            np.testing.assert_array_equal(old_p[0],new_p[0])
            self.assertGreater(expanded.encode(future)[0]['ids'][0],1)
            with patch.object(Provider,'get_word_vector',side_effect=AssertionError('cache reused')):
                self.assertEqual(subword.export_vectors(words,docs['fit'],docs['checkpoint'],Provider(),dict(sha256='fixture'),tmp)[1],record)

    def test_no_labels_used_and_cache_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs,words,rep,folder,record=self.representation(tmp)
            changed=[dict(d,bio=None,spans=[(0,1)]) for d in docs['checkpoint']]
            self.assertEqual(subword.export_vectors(words,docs['fit'],changed,None,dict(sha256='fixture'),tmp)[1],record)
            np.save(folder/'oov.npy',np.zeros((2,300),dtype=np.float32))
            with self.assertRaisesRegex(ValueError,'checksum'):subword.Representation(words,folder,record)

    def test_step_padding_masking_checkpoint_and_identical_resume(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(experiment,'Tracker',NoTracking):
            docs,words,rep,folder,record=self.representation(tmp)
            matrix=embeddings.matrices(words,Provider())[1]
            config=copy.deepcopy(experiment.CONFIG);config.update(epochs=2,batch_size=2)
            model=rep.build(matrix);encoded=rep.encode(docs['fit'][:1]);seq=encoded[0]['ids'];tags=encoded[0]['bio']
            model.eval()
            with torch.no_grad():
                a=model(torch.tensor([seq]),torch.tensor([len(seq)]))
                b=model(torch.tensor([seq+[0,0]]),torch.tensor([len(seq)]))
            torch.testing.assert_close(a,b[:,:len(seq)],rtol=0,atol=0)
            batch=collate_ner([(seq,tags),(seq[:1],tags[:1])])
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
            before=model.encoder.rnn.weight_ih_l0.detach().clone();vectors=model.encoder.embedding.oov.clone()
            values=experiment.base.update(model,batch,optimizer,torch.amp.GradScaler('cuda',enabled=False),torch.ones(3),'fp32')
            self.assertTrue(np.isfinite(values).all());self.assertFalse(torch.equal(before,model.encoder.rnn.weight_ih_l0))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            torch.testing.assert_close(vectors,model.encoder.embedding.oov,rtol=0,atol=0)
            self.assertEqual(float(model.encoder.embedding.weight.grad[0].abs().sum()),0)
            calls=[0]
            def interrupt(deadline):
                calls[0]+=1
                if calls[0]==2:raise experiment.BudgetEnd('fixture interruption')
            args=('fasttext_subword',docs,words,matrix)
            def run(name):
                return experiment.fit_arm(*args,Path(tmp)/name,'subword_fixture',record,tmp,'cpu',time.monotonic()+300,
                    config,encode=rep.encode,build=rep.build)
            with patch.object(experiment,'check_time',interrupt):
                with self.assertRaises(experiment.BudgetEnd):run('resumed')
            pointer=json.loads((Path(tmp)/'resumed/resume.json').read_text())
            self.assertEqual(torch.load(Path(tmp)/'resumed'/pointer['file'],weights_only=True)['progress']['batch'],1)
            resumed=run('resumed');full=run('full')
            a=torch.load(Path(tmp)/'resumed'/resumed['checkpoint'],weights_only=True)
            b=torch.load(Path(tmp)/'full'/full['checkpoint'],weights_only=True)
            self.assertTrue(all(torch.equal(a['model'][k],b['model'][k]) for k in a['model']))
            restored=rep.build(matrix);restored.load_state_dict(a['model']);restored.eval()
            model=rep.build(matrix);model.load_state_dict(b['model']);model.eval()
            with torch.no_grad():torch.testing.assert_close(restored(batch[0],batch[1]),model(batch[0],batch[1]),rtol=0,atol=0)
            with patch.object(rep,'build',side_effect=AssertionError('completed model retrained')):self.assertEqual(run('resumed'),resumed)

    def test_four_model_comparison_with_missing_csv_fields(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(experiment,'Tracker',NoTracking),patch.object(subword,'Tracker',NoTracking):
            docs,words,rep,table,record=self.representation(tmp)
            docs['checkpoint'][0]['spans']=[(0,4),(5,14)]
            docs['calibration']=[dict(docs['checkpoint'][0],row_id=101,group='calibration')]
            config=copy.deepcopy(experiment.CONFIG);config.update(epochs=1,batch_size=2)
            prepared=Path(tmp)/'prepared';prepared.mkdir();matrix=embeddings.matrices(words,Provider())[1]
            np.save(prepared/'fasttext.npy',matrix)
            vectors=dict(binary=dict(sha256='fixture'),files={'fasttext.npy':source.file_hash(prepared/'fasttext.npy')})
            spec=dict(content_sha256='fixture_split')
            common=dict(config=config,source_id='fixture_source',split='fixture_split',vocabulary=source.fingerprint(words),vectors=vectors)
            output=Path(tmp)/'results';source.save_json(output/'experiment_identity.json',common)
            choices={};rows=[];results={}
            with patch.object(experiment,'CONFIG',config):
                for arm in ['random','fasttext','fasttext_subword']:
                    folder=output/('subword' if arm=='fasttext_subword' else arm)
                    provenance=dict(subwords=record,vectors=vectors)
                    identity=source.fingerprint(dict(config=config,provenance=provenance)) if arm=='fasttext_subword' else source.fingerprint(dict(common,arm=arm))
                    results[arm]=experiment.fit_arm(arm,docs,words,matrix,folder,identity,provenance,tmp,'cpu',time.monotonic()+300,config,
                        encode=rep.encode if arm=='fasttext_subword' else None,build=rep.build if arm=='fasttext_subword' else None)
                    if arm=='fasttext_subword':
                        source.save_json(folder/'experiment_identity.json',dict(config=config,identity=identity,provenance=provenance))
                        source.save_json(folder/'selection_locked.json',dict(identity=identity,split='fixture_split',threshold=.8,
                            prediction_filter='strict',checkpoint_sha256=results[arm]['checkpoint_sha256']))
                        continue
                    choices[arm]=dict(sha256=results[arm]['checkpoint_sha256'])
                source.save_json(output/'selection_locked.json',dict(split='fixture_split',threshold=.8,prediction_filter='strict',checkpoints=choices))
                reference=experiment.build_model(matrix).eval()
                for arm in ['random','fasttext','reference']:
                    model=reference if arm=='reference' else experiment.load_arm(output/arm,words)[0]
                    sha=experiment.REFERENCE_SHA if arm=='reference' else results[arm]['checkpoint_sha256']
                    for part in ['checkpoint','calibration']:
                        encoded=source.encode_documents(docs[part],dict(kind='word',vocabulary=words))
                        p=experiment.prediction_cache(output/arm/(part+'.npz'),model,encoded,sha,
                            'reference_ngrams' if arm=='reference' else 'primary_word_unk')
                        rows.append(dict(model=arm,partition=part,**experiment.summarize(docs[part],p,words)))
                for row in rows:
                    row['fasttext_dictionary_coverage']=.9561 if row['model']=='fasttext' else np.nan
                    row['note']='official' if row['model']=='fasttext' else None
                pd.DataFrame(rows).to_csv(output/'comparison.csv',index=False)
                before=(output/'comparison.csv').read_bytes()
                with patch.object(experiment,'restore_model',return_value=(reference,{})),patch.object(experiment,'feature_documents',
                    side_effect=lambda ds,*args:source.encode_documents(ds,dict(kind='word',vocabulary=words))):
                    report=subword.evaluate_subword(docs,spec,words,prepared,table,record,output,'fixture_reference')
                self.assertEqual(len(report['metrics']),8);self.assertEqual(len(report['intervals']),6)
                encoded=json.dumps(report,allow_nan=False)
                self.assertEqual(json.loads(encoded),report)
                stored=json.loads((output/'subword/comparison.json').read_text())
                self.assertEqual(stored,report)
                for expected,actual in zip(rows,report['metrics']):
                    for name in ['f1','precision','recall']:
                        self.assertEqual(expected[name],actual[name])
                    if expected['model']!='fasttext':
                        self.assertIsNone(actual['fasttext_dictionary_coverage'])
                        self.assertIsNone(actual['note'])
                self.assertEqual(before,(output/'comparison.csv').read_bytes())
                self.assertTrue(report['protocol_precision_matches']);self.assertFalse(report['reference_replaced'])
                with self.assertRaisesRegex(ValueError,'verified OOV'):
                    experiment.load_arm(output/'subword',words)

    def test_official_and_gpu_guards(self):
        with self.assertRaisesRegex(ValueError,'verification'):
            subword.prepare_subwords({},[],[],{'source':{'url':'wrong'}},'/tmp','/tmp')
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'Colab CUDA'):
                subword.run_subword({},None,None,None,None,None,None,None,None,None)

    @unittest.skipUnless(importlib.util.find_spec('fasttext'),'Official SDK required')
    def test_actual_binary_generates_training_and_inference_oov(self):
        import fasttext
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'corpus.txt';path.write_text(('київ львів україна київський львівський\n')*100)
            # Zero-epoch SDK binary tests subword routing; NER optimization is separate.
            ft=fasttext.train_unsupervised(str(path),model='cbow',dim=300,minCount=1,epoch=0,lr=.001,
                bucket=1000,minn=5,maxn=5,thread=1,verbose=0,t=1.,loss='hs')
            binary=Path(tmp)/'tiny.bin';ft.save_model(str(binary));sha=source.file_hash(binary)
            loaded,_=embeddings.load_fasttext(binary,sha,official=False)
            docs,words=self.fixture()
            for word in ['рідкіснеслово','новеслово']:
                self.assertEqual(loaded.get_word_id(word),-1)
                self.assertGreater(np.linalg.norm(loaded.get_word_vector(word)),0)
            folder,record=subword.export_vectors(words,docs['fit'],docs['checkpoint'],loaded,dict(sha256=sha),tmp)
            rep=subword.Representation(words,folder,record);matrix=embeddings.matrices(words,loaded)[1]
            model=rep.build(matrix)
            for d in rep.encode(docs['fit'][:1]+docs['checkpoint']):
                actual=model.encoder.embedding(torch.tensor(d['ids'])).detach().numpy()
                for token,value in zip(d['tokens'],actual):
                    expected=embeddings.unit_rows(loaded.get_word_vector(experiment.token_key(token.value))[None])[0]
                    np.testing.assert_allclose(value,expected,rtol=1e-6,atol=1e-7)


if __name__=='__main__':unittest.main()
