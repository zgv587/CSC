# from torch.utils.data import DataLoader
#
# from config import *
# from data_processer.dataProcesser import CSCDataset
#
# if __name__ == '__main__':
#     ECSpell_law_train = CSCDataset(ECSpell_law_train_dir)
#     FASpell_ocr_train = CSCDataset(FASpell_ocr_train_dir)
#     NLPCC_TESTDATA = CSCDataset(NLPCC_TESTDATA_dir)
#
#     total_data = ECSpell_law_train + FASpell_ocr_train + NLPCC_TESTDATA
#
#     data_loader = DataLoader(total_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)
#
from transformers import BertTokenizer, BertModel
import torch

# 初始化 BERT tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")

# 示例输入句子
sentence = "我喜换学吸学用"
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)

# 获取预测标签
predictions = torch.argmax(outputs.logits, dim=-1)
