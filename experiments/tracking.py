"""Small W&B adapter extracted from the Part 2 notebook integration."""
import json
import os
from pathlib import Path
import shutil
import warnings

from part2 import rng_state, restore_rng
from part2_data import fingerprint, save_json


class Tracker:
    def __init__(self, folder, config, mode=None):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.config=config;self.run=None
        self.mode=mode or ('online' if os.environ.get('WANDB_API_KEY') else 'offline')
        self.run_id=fingerprint(config['identity'])[:16]
        self.last=0
        state=rng_state()
        try:
            import wandb
            self.sdk=wandb
            args=dict(entity='dtkach-kse',project='ai301-a3',id=self.run_id,
                name=config['name'],group='fasttext_v1',config=config,save_code=False,
                dir=str(self.folder),settings=wandb.Settings(init_timeout=12,finish_timeout=10,console='off',disable_git=True))
            try:
                self.run=wandb.init(**args,mode=self.mode,**({'resume':'allow'} if self.mode=='online' else {}))
            except Exception as error:
                warnings.warn(f'W&B {self.mode} init failed ({type(error).__name__}); using offline logs')
                wandb.finish(exit_code=1);self.mode='offline';self.run=wandb.init(**args,mode='offline')
            self.run.define_metric('epoch');self.run.define_metric('*',step_metric='epoch')
            index=self.folder/'wandb_index.json'
            if self.mode=='online' and index.exists():self.last=json.loads(index.read_text()).get('online_epoch',0)
        except Exception as error:
            warnings.warn(f'W&B unavailable ({type(error).__name__}); checkpoints and CSV remain authoritative')
        finally:restore_rng(state)

    def log(self, row):
        if self.run is None:return
        state=rng_state()
        try:
            epoch=int(row['epoch'])
            if epoch<=self.last:return
            values={k:v for k,v in row.items() if isinstance(v,(float,int))}
            if isinstance(row.get('gradient_flow'),str):
                values.update({'gradient_flow/'+k:v for k,v in json.loads(row['gradient_flow']).items()})
            self.run.log(values);self.last=epoch
        except Exception as error:warnings.warn(f'W&B log failed ({type(error).__name__}); CSV retained')
        finally:restore_rng(state)

    def finish(self, summary, images=()):
        if self.run is None:return
        state=rng_state()
        try:
            self.run.summary.update(summary)
            for path in images:
                if Path(path).exists():self.run.log({'plots/'+Path(path).stem:self.sdk.Image(str(path))})
            self.run.finish()
            save_json(self.folder/'wandb_index.json',dict(id=self.run_id,mode=self.mode,
                directory=str(Path(self.run.dir).parent),online_epoch=self.last if self.mode=='online' else 0,
                note='Sync only latest offline snapshot for this run ID'))
        except Exception as error:warnings.warn(f'W&B finish failed ({type(error).__name__}); local results retained')
        finally:restore_rng(state)
