"""Character n-gram word representation used by the completed A3 run.

No vocabulary is trained or updated on evaluation/test texts.  They only
receive deterministic routing IDs to frozen word and n-gram embeddings.
"""

import hashlib

import torch
from torch import nn

from ner_data import token_key
from recurrent_model import NERModel


ENCODER = dict(rnn_type="lstm", num_layers=2, embedding_dim=64,
               hidden_size=64, dropout=.2, bidirectional=True)


def ngrams(word, buckets):
    word = "<" + word + ">"
    return ([1 + int.from_bytes(hashlib.sha256(word[i:i + width].encode()).digest()[:8], "big")
             % (buckets - 1)
             for width in range(3, 6)
             for i in range(len(word) - width + 1)] or [1])


class WordFeatures(nn.Module):
    """Byte-stable 3–5-character n-grams with word lookup residual."""

    def __init__(self, state, word_embedding):
        super().__init__()
        if state["kind"] != "ngrams":
            raise ValueError("Only the selected ngrams candidate is supported")
        self.kind = "ngrams"
        self.word = word_embedding
        self.subword = nn.EmbeddingBag(state["pieces"], 64, mode="mean", padding_idx=0)
        for name in ("word_ids", "begin", "counts", "flat"):
            self.register_buffer(name, torch.zeros(1, dtype=torch.long), persistent=False)

    def set_table(self, table):
        device = self.subword.weight.device
        for name, value in table.items():
            setattr(self, name, torch.tensor(value, dtype=torch.long, device=device))

    def forward(self, ids):
        unique, inverse = torch.unique(ids.reshape(-1), return_inverse=True)
        counts = self.counts[unique]
        offsets = torch.cumsum(counts, 0) - counts
        positions = (torch.repeat_interleave(self.begin[unique] - offsets, counts)
                     + torch.arange(int(counts.sum()), device=ids.device))
        values = self.subword(self.flat[positions], offsets)
        values = values + self.word(self.word_ids[unique])
        return values[inverse].reshape(*ids.shape, 64)


def new_model(state):
    model = NERModel(state["size"], **ENCODER)
    model.encoder.embedding = WordFeatures(state, model.encoder.embedding)
    return model


def feature_documents(docs, state, model):
    """Build non-learned routing table for a batch of original-offset docs."""
    keys = {}
    word_ids, begin, counts, flat = [0], [0], [1], [0]
    result = []
    for doc in docs:
        ids = []
        for token in doc["tokens"]:
            key = token_key(token.value)
            if key not in keys:
                keys[key] = len(word_ids)
                pieces = ngrams(key, state["buckets"])
                word_ids.append(state["vocabulary"].get(key, 1))
                begin.append(len(flat))
                counts.append(len(pieces))
                flat.extend(pieces)
            ids.append(keys[key])
        result.append(dict(doc, ids=ids))
    model.encoder.embedding.set_table(dict(word_ids=word_ids, begin=begin,
                                            counts=counts, flat=flat))
    return result


def restore_model(checkpoint, expected_sha256, split_hash, device="cuda"):
    from part2_data import file_hash

    if file_hash(checkpoint) != expected_sha256:
        raise ValueError("Selected character n-gram checkpoint hash changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["split"] != split_hash:
        raise ValueError("Selected model was trained on another frozen split")
    state = saved["representation"]
    if state["kind"] != "ngrams" or state["buckets"] != 16384:
        raise ValueError("Unexpected representation hyperparameters")
    model = new_model(state)
    model.load_state_dict(saved["model"], strict=True)
    model.to(device)
    return model, state
