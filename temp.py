# import torch
# import numpy as np
# import torch.nn as nn
# from torch import optim
# from torch.utils.data import DataLoader
# from transformers import BertTokenizer, BertModel, DataCollatorWithPadding, AutoTokenizer

# from config import *
# from CSC_model import CSCModel
# from data_processer.dataProcesser import CSCDataset

# if __name__ == '__main__':
#     # 加载预训练模型
#     tokenizer = BertTokenizer.from_pretrained(checkpoint)
#     roberta = BertModel.from_pretrained(checkpoint)

#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#     ECSpell_law_train = CSCDataset(ECSpell_law_train_dir, tokenizer)
#     # FASpell_ocr_train = CSCDataset(FASpell_ocr_train_dir, tokenizer)
#     # NLPCC_TESTDATA = CSCDataset(NLPCC_TESTDATA_dir, tokenizer)

#     # total_data = ECSpell_law_train + FASpell_ocr_train + NLPCC_TESTDATA
#     total_data = ECSpell_law_train

#     data_loader = DataLoader(
#         dataset=total_data,
#         num_workers=num_workers,
#         batch_size=batch_size,
#         shuffle=True,
#         # collate_fn=data_collator  # map-style 时使用
#     )

#     model = CSCModel(max_length)
#     model.cuda()

#     criterion = nn.CrossEntropyLoss()

#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#     def train(epoch):
#         model.train()
#         train_loss = 0
#         for data, mask, label in data_loader:
#             data, mask, label = data.cuda(), mask.cuda(), label.cuda()
#             optimizer.zero_grad()
#             output = model(data, mask)
#             loss = criterion(output, label)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * data.size(0)
#         train_loss = train_loss / len(data_loader.dataset)
#         print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


#     # def val(epoch):
#     #     model.eval()
#     #     val_loss = 0
#     #     gt_labels = []
#     #     pred_labels = []
#     #     with torch.no_grad():
#     #         for data, label in test_loader:
#     #             data, label = data.cuda(), label.cuda()
#     #             output = model(data)
#     #             preds = torch.argmax(output, 1)
#     #             gt_labels.append(label.cpu().data.numpy())
#     #             pred_labels.append(preds.cpu().data.numpy())
#     #             loss = criterion(output, label)
#     #             val_loss += loss.item() * data.size(0)
#     #     val_loss = val_loss / len(test_loader.dataset)
#     #     gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
#     #     acc = np.sum(gt_labels == pred_labels) / len(pred_labels)
#     #     print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))


#     for epoch in range(1, epochs + 1):
#         train(epoch)
#         # val(epoch)

from data_processer import CSCDataset
print(1)