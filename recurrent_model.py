"""Shared recurrent encoder for the NER baseline and adversarial validation.

No attention, pretrained transformer, or convolution is used.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurrentEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=64, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.output_dim = hidden_size * 2

    def forward(self, token_ids, lengths):
        embeddings = self.dropout(self.embedding(token_ids))
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.rnn(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=token_ids.size(1))
        return self.dropout(encoded)


class NERModel(nn.Module):
    """BIO token classification head; the vocabulary and offsets are external."""
    def __init__(self, vocab_size, num_tags=3, **encoder_kwargs):
        super().__init__()
        self.encoder = RecurrentEncoder(vocab_size, **encoder_kwargs)
        self.classifier = nn.Linear(self.encoder.output_dim, num_tags)

    def forward(self, token_ids, lengths):
        return self.classifier(self.encoder(token_ids, lengths))


class DomainModel(nn.Module):
    """Binary source/test head on the same encoder implementation/configuration."""
    def __init__(self, vocab_size, **encoder_kwargs):
        super().__init__()
        self.encoder = RecurrentEncoder(vocab_size, **encoder_kwargs)
        self.classifier = nn.Linear(2*self.encoder.output_dim, 1)

    def forward(self, token_ids, lengths):
        states = self.encoder(token_ids, lengths)
        mask = torch.arange(token_ids.size(1), device=token_ids.device)[None, :] < lengths[:, None]
        mean = (states * mask[:, :, None]).sum(1) / lengths[:, None]
        maximum = states.masked_fill(~mask[:, :, None], float('-inf')).amax(1)
        return self.classifier(torch.cat([mean, maximum], dim=1)).squeeze(-1)
