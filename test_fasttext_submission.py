import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import fasttext_submission as submission
import part2
import part2_data as source
from pilot_ner import documents
from experiments import fasttext_subword as subword
from experiments import fasttext_experiment as experiment
from experiments import fasttext_embeddings as embeddings
from test_fasttext import Provider
import test_fasttext_subword as fixtures


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_missing_only_and_immutable_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs,words,rep,table,record=fixtures.SubwordTests().representation(tmp)
            old={p.name:p.read_bytes() for p in table.iterdir()}
            new_docs=documents(pd.DataFrame(dict(text=['Київ рідкіснеслово новітнємісто'],loc_markers=['[]'])))
            calls=[]
            class Spy(Provider):
                def get_word_vector(self,key):calls.append(key);return super().get_word_vector(key)
            with patch.object(submission,'verified_provider',return_value=Spy()):
                folder,extended,count=submission.extend_vectors(words,new_docs,record,dict(binary=dict(sha256='fixture')),Path(tmp),tmp)
            self.assertEqual(calls,['новітнємісто']);self.assertEqual(count,1)
            self.assertEqual({p.name:p.read_bytes() for p in table.iterdir()},old)
            keys,vectors=subword.verify_vectors(folder,extended,words)
            self.assertIn('новітнємісто',keys)
            model=rep.build(embeddings.matrices(words,Provider())[1]);saved=dict(model=model.state_dict())
            result=dict(checkpoint_sha256='fixture_checkpoint',initial_recurrent_hash=experiment.recurrent_hash(model))
            for name in ['checkpoint','calibration']:
                if name=='calibration':docs[name]=[dict(docs['checkpoint'][0],row_id=101,group='calibration')]
                encoded=rep.encode(docs[name])
                experiment.prediction_cache(Path(tmp)/(name+'.npz'),model,encoded,result['checkpoint_sha256'],
                    'trained_fasttext_oov_'+source.fingerprint(record))
            expanded=subword.Representation(words,folder,extended)
            loaded,checks=submission.check_old_predictions(docs,rep,expanded,saved,result,Path(tmp),torch.device('cpu'))
            self.assertTrue(all(r['expanded_predictions_bit_identical'] for r in checks.values()))
            with patch.object(submission,'verified_provider',side_effect=AssertionError('cache hit must not load binary')):
                self.assertEqual(submission.extend_vectors(words,new_docs,record,dict(binary=dict(sha256='fixture')),Path(tmp),tmp)[1],extended)
                self.assertEqual(submission.extend_vectors(words,docs['checkpoint'],record,dict(binary=dict(sha256='fixture')),Path(tmp),tmp)[1],record)
            values=np.load(folder/'oov.npy');values[keys.index('рідкіснеслово'),0]+=1
            np.save(folder/'oov.npy',values)
            extended['files']['oov.npy']=source.file_hash(folder/'oov.npy');source.save_json(folder/'vectors.json',extended)
            with self.assertRaisesRegex(ValueError,'Old OOV vectors changed'):
                submission.extend_vectors(words,new_docs,record,dict(binary=dict(sha256='fixture')),Path(tmp),tmp)

    def test_pinned_source_copy_and_rejection(self):
        import hashlib
        import sys
        original_path=list(sys.path)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);folder=root/'fasttext_subword_source';folder.mkdir()
                code=b'value = 1\n';(folder/'tiny.py').write_bytes(code)
                files={'tiny.py':hashlib.sha256(code).hexdigest()}
                source.save_json(folder/'source_manifest.json',dict(files=files,source_id=source.fingerprint(files)))
                with patch.object(submission,'MANIFEST_SHA',source.file_hash(folder/'source_manifest.json')):
                    copied,identity=submission.verified_source(root,root/'work')
                    self.assertEqual(identity,source.fingerprint(files))
                    self.assertEqual((copied/'tiny.py').read_bytes(),code)
                    (folder/'tiny.py').write_text('changed')
                    with self.assertRaisesRegex(ValueError,'source changed'):submission.verified_source(root,root/'work')
                    (folder/'source_manifest.json').write_text('{}')
                    with self.assertRaisesRegex(ValueError,'manifest changed'):submission.verified_source(root,root/'work')
        finally:sys.path[:]=original_path

    def test_historical_manifest_pin(self):
        manifest=Path('fasttext_source_manifest.json')
        self.assertEqual(source.file_hash(manifest),submission.MANIFEST_SHA)
        record=json.loads(manifest.read_text())
        self.assertEqual(record['source_id'],'9b66382ec62e6c220b15f36c9104c79cf70b8233cd081e0c6e276c1bfeee98bd')
        self.assertEqual(record['source_id'],source.fingerprint(record['files']))

    def test_numerical_drift_keeps_exact_entities_and_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs,words,rep,table,record=fixtures.SubwordTests().representation(tmp)
            docs['calibration']=[dict(docs['checkpoint'][0],row_id=101,group='calibration')]
            model=rep.build(embeddings.matrices(words,Provider())[1])
            saved=dict(model=model.state_dict());result=dict(checkpoint_sha256='fixture',initial_recurrent_hash=experiment.recurrent_hash(model))
            for label,current,old,error in [('small',.9,.9004,None),('large',.9,.9006,AssertionError),
                                             ('boundary',.7998,.8002,ValueError)]:
                folder=Path(tmp)/label
                cached=[np.array([[1-old,old,0.],[1.,0.,0.]])]
                predicted=[np.array([[1-current,current,0.],[1.,0.,0.]])]
                with patch.object(experiment,'infer_probabilities',return_value=cached):
                    for part in ['checkpoint','calibration']:
                        experiment.prediction_cache(folder/(part+'.npz'),model,rep.encode(docs[part]),result['checkpoint_sha256'],
                            'trained_fasttext_oov_'+source.fingerprint(record))
                with patch.object(experiment,'infer_probabilities',return_value=predicted):
                    if error:
                        with self.assertRaises(error):submission.check_old_predictions(docs,rep,rep,saved,result,folder,torch.device('cpu'))
                    else:submission.check_old_predictions(docs,rep,rep,saved,result,folder,torch.device('cpu'))

    def test_locked_checkpoint_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs,words,rep,table,record=fixtures.SubwordTests().representation(tmp);folder=Path(tmp)
            matrix=embeddings.matrices(words,Provider())[1];model=rep.build(matrix)
            spec=dict(content_sha256='split',files={'test.csv':'hash'},fit=[1,2])
            provenance=dict(source_id='source',split='split',data_files=spec['files'],fit_ids=source.fingerprint(spec['fit']),vocabulary=source.fingerprint(words))
            identity=source.fingerprint(dict(config=experiment.CONFIG,provenance=provenance))
            source.save_json(folder/'experiment_identity.json',dict(identity=identity,config=experiment.CONFIG,provenance=provenance))
            torch.save(dict(identity=identity,arm='fasttext_subword',epoch=6,encoder=experiment.CONFIG['encoder'],vocabulary=words,model=model.state_dict()),folder/'model.pt')
            sha=source.file_hash(folder/'model.pt')
            result=dict(complete=True,name='fasttext_subword',identity=identity,checkpoint='model.pt',checkpoint_sha256=sha,best_epoch=6)
            lock=dict(identity=identity,checkpoint_sha256=sha,split='split',threshold=.8,prediction_filter='strict')
            source.save_json(folder/'result.json',result);source.save_json(folder/'selection_locked.json',lock)
            self.assertEqual(submission.locked_checkpoint(folder,spec,'source')[0]['vocabulary'],words)
            source.save_json(folder/'selection_locked.json',dict(lock,threshold=.7))
            with self.assertRaisesRegex(ValueError,'Prediction policy'):submission.locked_checkpoint(folder,spec,'source')
            source.save_json(folder/'selection_locked.json',dict(lock,checkpoint_sha256='wrong'))
            with self.assertRaisesRegex(ValueError,'SHA256'):submission.locked_checkpoint(folder,spec,'source')
            source.save_json(folder/'selection_locked.json',lock)
            (folder/'model.pt').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'SHA256'):submission.locked_checkpoint(folder,spec,'source')

    def test_export_477_order_json_offsets_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            text='Івано-Франківської області 😀'
            docs=documents(pd.DataFrame(dict(text=[text]*477,loc_markers=['[(0,26)]']*477)))
            for i,d in enumerate(docs):d['text_id']=477-i
            probabilities=[np.eye(3)[d['bio']] for d in docs]
            path=Path(tmp)/'local.csv';part2.export_submission(docs,probabilities,path,.8,'strict')
            csv=pd.read_csv(path);self.assertEqual(csv.text_id.tolist(),list(range(477,0,-1)))
            self.assertEqual(csv.columns.tolist(),['text_id','locations']);self.assertEqual(csv.text_id.nunique(),477)
            self.assertTrue(all(json.loads(v)==[text[:26]] for v in csv.locations))
            target=Path(tmp)/'persistent.csv';submission.write_once(path,target)
            before=target.stat().st_mtime_ns;submission.write_once(path,target);self.assertEqual(before,target.stat().st_mtime_ns)
            path.write_text('different')
            with self.assertRaises(FileExistsError):submission.write_once(path,target)
            self.assertEqual(csv.to_csv(index=False),target.read_text())


if __name__=='__main__':unittest.main()
