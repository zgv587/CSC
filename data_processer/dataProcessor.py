from typing import *

import torch
import tqdm
from config import *
from torch.utils.data import Dataset, random_split
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


def split_torch_dataset(dataset, test_size=0.2, random_seed=None):
    """
    将PyTorch Dataset对象划分为训练集和测试集。

    参数:
        dataset (Dataset): PyTorch的Dataset对象。
        test_size (float or int): 如果是浮点数，表示测试集相对于整个数据集的比例；如果是整数，表示测试集的样本数量。默认值为0.2。
        random_seed (int or None): 随机种子，用于控制随机性以确保结果可复现。默认值为None。

    返回:
        tuple: 一个包含两个Dataset对象的元组 (train_dataset, test_dataset)。
    """
    # 计算训练集和测试集的大小
    total_length = len(dataset)
    if isinstance(test_size, float):
        test_length = int(total_length * test_size)
    else:
        test_length = test_size
    train_length = total_length - test_length

    # 设置随机种子
    generator = (
        torch.Generator().manual_seed(random_seed) if random_seed is not None else None
    )

    # 分割数据集
    train_dataset, test_dataset = random_split(
        dataset, [train_length, test_length], generator=generator
    )

    return train_dataset, test_dataset
