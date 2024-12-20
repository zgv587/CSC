# 配置文件

# confusion set
confusion_set_path = "./data/confusion.txt"

# SIGHAN数据
SIGHAN_train_dir_corr = "./data/traditional/train15_correct.txt"
SIGHAN_train_dir_err = "./data/traditional/train15_error.txt"

SIGHAN_train_dir_corr14 = "./data/traditional/train14_correct.txt"
SIGHAN_train_dir_err14 = "./data/traditional/train14_error.txt"

# 苏大数据，格式为 错误数 原句 正确句
ECSpell_law_train_dir = "./data/ECSPell/law.train"
ECSpell_law_test_dir = "./data/ECSPell/law.test"
ECSpell_med_train_dir = "./data/ECSPell/med_train"
ECSpell_med_test_dir = "./data/ECSPell/med_test"
ECSpell_odw_train_dir = "./data/ECSPell/odw_train"
ECSpell_odw_test_dir = "./data/ECSPell/odw_test"

# 爱奇艺数据，格式为 错误数 原句 正确句
FASpell_char_meta_dir = "./data/FASpell_data/char_meta"
FASpell_ocr_train_dir = "./data/FASpell_data/ocr_train_3575.txt"
FASpell_ocr_test_dir = "./data/FASpell_data/ocr_test_100.txt"

# NLPCC2023 task8 数据， 格式为 原句 正确句
NLPCC_TESTDATA_dir = "./data/NLPCC2023_Shared_Task8-main/data.txt"

epochs = 50  # 迭代次数

exit_ep = 10  # 早退轮数

num_workers = 4  # dataloader的num_workers

batch_size = 4  # 批次大小

learning_rate = 5e-5  # 学习率

decay_rate = 0.9  # 学习率衰减速率

shuffle = True  # 是否打乱数据集

checkpoint = "bert-base-chinese"  # 预训练模型

max_length = 256  # padding的最大长度
