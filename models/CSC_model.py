import torch
import torch.nn as nn
from torch.nn import Embedding, LayerNorm
from transformers import BertModel, BertTokenizer

from config import *
from utils import beam_search_generate
from data_processer import load_confusion


class CombineBertModel(nn.Module):
    def __init__(
        self,
        encoder_model,
        decoder_model,
        confusion_set=None
    ):
        super(CombineBertModel, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.confusion_set = confusion_set

        self.linear = nn.Linear(
            decoder_model.config.hidden_size, encoder_model.config.vocab_size
        )

    def forward(self, src, src_mask):
        x = self.encoder(src, attention_mask=src_mask).last_hidden_state
        x = self.decoder(x)

        x = self.linear(x)
        if confusion_set:
            x = self.limit_with_confusion(src, x)

        return x

    def limit_with_confusion(
        self, src, outputs, confusion_value=0.95, out_of_confusion=0.05, pad_token_id=0
    ):
        assert (
            len(outputs.shape) == 3
        ), "the sequence dim must be 3"  # (batch_size, seq_len, vacob_size)
        m = torch.ones_like(outputs)
        for i in range(src.shape[1]):
            charindexs = src[:, i]
            if charindexs.eq(pad_token_id).all():
                break
            for char in charindexs:
                if char in self.confusion_set:
                    char_confusion = self.confusion_set[char]
                    values = torch.full((vocab_size,), out_of_confusion)
                    values[char_confusion] = confusion_value
                    m[:, i, :] = values
        return outputs.mul(m)

    generate_with_beam = beam_search_generate

    def save(self, store_path):
        torch.save(self, store_path)

    def save_state(self, store_path):
        torch.save(self.state_dict(), store_path)
