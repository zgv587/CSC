import sys
import tqdm 
from typing import Union, List

from config import *

'''
class DataPreprocessor:
    def __init__(self, 
                 path: Union[str, List[str]], 
                 tokenizer: object
                 ):
        self.path = path
        self.tokenizer = tokenizer
        self.data = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "labels_ids": []
        }

        self.preprocess()

    def preprocess(self):
        if isinstance(self.path, list):
            self.handle_sighan()

    def handle_sighan(self):
        x, y = self.path 
        cnt = 0
        with open(x, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f, desc="preprocessing sighan dataset"):
                line = line.strip()
                cnt += 1

                data_tokenizers = self.tokenizer(
                    line,
                    padding='max_length',
                    max_length=max_length
                )

                for key, value in data_tokenizers.items():
                    self.data[key].append(value)

        with open(y, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f, desc="preprocessing sighan dataset", total=cnt):
                line = line.strip()
                self.data["labels_ids"].append(
                    self.tokenizer(line, padding='max_length', max_length=max_length)[
                        "input_ids"
                        ]
                )


        return data_tokenizers

    # def preprocess(self, path):
        # with open(path, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip()
        #         new_line = line.split()

        #         data_tokenizers = self.tokenizer(
        #             new_line[-2],
        #             padding='max_length',
        #             max_length=max_length
        #         )
        #         data_tokens, data_mask = data_tokenizers['input_ids'], data_tokenizers['attention_mask']

        #         self.data.append(
        #             data_tokens
        #         )

        #         self.mask.append(
        #             data_mask
        #         )

        #         self.labels.append(
        #             self.tokenizer(
        #                 new_line[-1],
        #                 padding='max_length',
        #                 max_length=max_length
        #             )['input_ids']
        #         )


class ECSpellPreprocessor:
    def __init__(self, path):
        pass


class FASpellPreprocessor:
    def __init__(self, path):
        pass


class NLPcc2023Preprocessor:
    def __init__(self, path):
        pass


class SIGHANPreprocessor:
    def __init__(self, path):
        pass
'''

