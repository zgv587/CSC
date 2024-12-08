from transformers import BertTokenizer, BertModel 

from config import *
from data_processer.preprocessing import *

#加载预训练字典和分词方法
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',  # 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese
)

bert = BertModel.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese'
)

train_dataset = DataPreprocessor(
    path = [SIGHAN_train_dir_corr, SIGHAN_train_dir_err],
    tokenizer = tokenizer
)

print(train_dataset.data)