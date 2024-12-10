from typing import *

import torch
import tqdm
from config import *
from torch.utils.data import Dataset
from transformers import BertTokenizer

"""
class CSCDataset(Dataset):
    def __init__(
            self,
            path: Union[str, List[str]],  # path 数据集路径
            tokenizer: object,  # tokenizer 分词器，text2id
    ):
        # assert len(data) == len(label)
        self.path = path
        self.tokenizer = tokenizer

        self.data_processor()

    def data_processor(self):
        \"""
        DataPreprocessing, generate data and labels for CSC dataset. And convert text to tensors.
        :return: None
        \"""
        data_preprocessor = DataPreprocessor(
            path=self.path,
            tokenizer=self.tokenizer
        )

        self.data = data_preprocessor.data.values()
        self.length = len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index])

    def __len__(self):
        return self.length
"""


class CSCDataset(Dataset):
    def __init__(
        self,
        path: Union[str, List[str]],  # path 数据集路径
        tokenizer: object,  # tokenizer 分词器，text2id
    ):
        # assert len(data) == len(label)
        self.path = path
        self.tokenizer = tokenizer

        self.raw_sentences = []
        self.corr_sentences = []

        self.data = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "labels_ids": [],
        }

        self.lines_num = 0
        self.words_num = 0
        self.max_length = 0

        self.INIT()

    def INIT(self):
        self.data_processor()
        print(
            f"共{self.lines_num}句，共{self.words_num}字，最长的句子有{self.max_length}字"
        )
        self.max_length = 512

    def data_processor(self):
        if isinstance(self.path, list):
            self.handle_sighan()
        elif isinstance(self.path, str):
            pass

    def handle_sighan(self):
        xpath, ypath = self.path
        with open(xpath, "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f, desc="preprocessing sighan dataset"):
                line = line.strip()
                self.lines_num += 1
                self.words_num += len(line)
                self.max_length = max(self.max_length, len(line))
                self.raw_sentences.append(line)

        with open(ypath, "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(
                f, desc="preprocessing sighan dataset", total=self.lines_num
            ):
                line = line.strip()
                self.corr_sentences.append(line)

    def __len__(self):
        return len(self.raw_sentences)

    def __getitem__(self, idx):
        raw_sentence = self.raw_sentences[idx]
        corr_sentence = self.corr_sentences[idx]

        encoding = self.tokenizer.encode_plus(
            raw_sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        labels = self.tokenizer.encode(
            corr_sentence,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).flatten()

        # input_ids = encoding["input_ids"]
        # attention_mask = encoding["attention_mask"]

        # labels = self.tokenizer.encode(corr_sentence)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
