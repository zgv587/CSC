import torch
import torch.nn as nn
from torch.nn import Embedding, LayerNorm
from transformers import BertModel, BertTokenizer

from config import *


class CombineBertModel(nn.Module):
    def __init__(
        self,
        encoder_model,
        decoder_model,
    ):
        super(CombineBertModel, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.is_transformer = self.decoder.config.is_transformer

        if not self.is_transformer:
            self.linear = nn.Linear(
                decoder_model.config.hidden_size, encoder_model.config.vocab_size
            )

    def forward(self, src, src_mask):
        x = self.encoder(src, attention_mask=src_mask).last_hidden_state
        x = self.decoder(x)

        if not self.is_transformer:
            x = self.linear(x)

        return x
