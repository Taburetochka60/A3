import hashlib
import json
import torch
from torch import nn
from tokenizers import Tokenizer,models,trainers
import part2 as base
import part2_data as source
from ner_data import token_key
from pilot_ner import vocabulary
from recurrent_model import NERModel
POLICY={"buckets":16384}
write_json=source.save_json

def fit_representation(train, kind, folder):
    train_ids = source.fingerprint([d['row_id'] for d in train])
    path = folder/'representation.json'
    if path.exists():
        state = json.loads(path.read_text())
        assert state['train_ids'] == train_ids and state['kind'] == kind
        return state
    words = vocabulary(train)
    state = dict(kind=kind, vocabulary=words, size=len(words)+2, train_ids=train_ids,
                 buckets=POLICY['buckets'])
    if kind == 'bpe':
        tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
        trainer = trainers.BpeTrainer(vocab_size=30000, min_frequency=2,
                    special_tokens=['[PAD]','[UNK]'], show_progress=False)
        tokenizer.train_from_iterator((token_key(t.value) for d in train for t in d['tokens']), trainer=trainer)
        state.update(tokenizer=tokenizer.to_str(), pieces=tokenizer.get_vocab_size())
    else:
        assert kind == 'ngrams'
        state['pieces'] = POLICY['buckets']
    write_json(path,state)
    return state

def ngrams(word, buckets):
    word = '<'+word+'>'
    return [1+int.from_bytes(hashlib.sha256(word[i:i+n].encode()).digest()[:8],'big')%(buckets-1)
            for n in range(3,6) for i in range(len(word)-n+1)] or [1]

class WordFeatures(nn.Module):
    def __init__(self, state, word_embedding):
        super().__init__()
        self.kind = state['kind']
        self.word = word_embedding if self.kind == 'ngrams' else None
        self.subword = nn.EmbeddingBag(state['pieces'], 64, mode='mean', padding_idx=0)
        for name in ['word_ids','begin','counts','flat']:
            self.register_buffer(name, torch.zeros(1,dtype=torch.long), persistent=False)

    def set_table(self, table):
        device = self.subword.weight.device
        for name, value in table.items():
            setattr(self,name,torch.tensor(value,dtype=torch.long,device=device))

    def forward(self, ids):
        unique, inverse = torch.unique(ids.reshape(-1),return_inverse=True)
        counts = self.counts[unique]
        offsets = torch.cumsum(counts,0)-counts
        positions = (torch.repeat_interleave(self.begin[unique]-offsets,counts)
                     + torch.arange(int(counts.sum()),device=ids.device))
        values = self.subword(self.flat[positions],offsets)
        if self.word is not None:
            values = values+self.word(self.word_ids[unique])
        return values[inverse].reshape(*ids.shape,64)

def feature_documents(docs, state, model):
    # Routing IDs do not create trainable word embeddings for heldout words.
    keys = {}; word_ids=[0]; begin=[0]; counts=[1]; flat=[0]
    tokenizer = Tokenizer.from_str(state['tokenizer']) if state['kind']=='bpe' else None
    encoded=[]
    for d in docs:
        ids=[]
        for token in d['tokens']:
            key=token_key(token.value)
            if key not in keys:
                keys[key]=len(word_ids)
                pieces=(tokenizer.encode(key,add_special_tokens=False).ids if tokenizer else ngrams(key,state['buckets'])) or [1]
                word_ids.append(state['vocabulary'].get(key,1))
                begin.append(len(flat));counts.append(len(pieces));flat.extend(pieces)
            ids.append(keys[key])
        encoded.append(dict(d,ids=ids))
    model.encoder.embedding.set_table(dict(word_ids=word_ids,begin=begin,counts=counts,flat=flat))
    return encoded

def new_model(state):
    model=NERModel(state['size'],**base.CONFIG['encoder'])
    model.encoder.embedding=WordFeatures(state,model.encoder.embedding)
    return model


def restore_model(checkpoint, expected_sha256, split_hash, device='cpu'):
    if source.file_hash(checkpoint)!=expected_sha256:raise ValueError('BPE checkpoint checksum changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['split']!=split_hash or saved['representation']['kind']!='bpe':raise ValueError('BPE split or representation differs')
    model=new_model(saved['representation']);model.load_state_dict(saved['model'],strict=True)
    return model.to(device).eval(),saved['representation']
