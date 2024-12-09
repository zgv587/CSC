import torch
import torch.nn as nn
from torch.nn import Embedding, LayerNorm
from transformers import BertModel, BertTokenizer

from config import *

from .transformers_layer import TransformerDecoder, TransformerDecoderLayer


class CSCModel(nn.Module):
    def __init__(self, output_dim):
        super(CSCModel, self).__init__()
        self.bert = BertModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(0.1)
        self.correct = nn.Linear(
            self.bert.config.hidden_size, self.bert.config.vocab_size
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]


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


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        model,
        nhead,
        num_decoder_layers,
        dim_feedforward,
        max_seq_len,
        dropout=0.1,
    ):
        super(Seq2SeqModel, self).__init__()
        self.model = model

        self.embedding = Embedding(model.config.vocab_size, model.config.hidden_size)
        self.positional_encoding = self._generate_positional_encoding(
            max_seq_len, model.config.hidden_size
        ).to(torch.device("cuda"))
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(
                model.config.hidden_size, nhead, dim_feedforward, dropout
            ),
            num_decoder_layers,
            LayerNorm(model.config.hidden_size),
        )
        self.num_decoder_layers = num_decoder_layers
        self.fc_out = nn.Linear(model.config.hidden_size, model.config.vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _generate_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
    ):
        encoder_outputs = self.model(src, attention_mask=src_mask)[0]
        tgt_embedded = self.embedding(src) + self.positional_encoding[: src.size(0), :]
        tgt_embedded = self.dropout(tgt_embedded)
        decoder_outputs = self.decoder(
            tgt_embedded,
            tgt_key_padding_mask=src_mask,
        )
        output = self.fc_out(decoder_outputs)
        return output


# class CSCModel(nn.Module):
#     def __init__(self, input_size):
#         super(CSCModel, self).__init__()
#         self.bert = BertModel.from_pretrained(checkpoint, trust_remote_code=True)
#         self.fc = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, input_size),
#             # nn.Sigmoid()
#         )

#     def forward(self, x, mask):
#         x = x * mask
#         x = self.fc(x)
#         return x
