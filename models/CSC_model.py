import torch
import torch.nn as nn
from torch.nn import Embedding, LayerNorm
from transformers import BertModel, BertTokenizer

from config import *
from utils import beam_search_generate


class CombineBertModel(nn.Module):
    def __init__(
        self,
        encoder_model,
        decoder_model,
    ):
        super(CombineBertModel, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model

        self.linear = nn.Linear(
            decoder_model.config.hidden_size, encoder_model.config.vocab_size
        )

    def forward(self, src, src_mask):
        x = self.encoder(src, attention_mask=src_mask).last_hidden_state
        x = self.decoder(x)

        x = self.linear(x)

        return x

    generate_with_beam = beam_search_generate

    def save(self, store_path):
        torch.save(self, store_path)

    def save_state(self, store_path):
        torch.save(self.state_dict(), store_path)
