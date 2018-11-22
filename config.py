# -*- coding: utf-8 -*-

config = {
    'sequence_length': 300,    # 文本长度，当文本大于该长度则截断
    'num_classes': 10,         # 文本分类数
    'embedding_size': 300,     # 字向量维度
    'vocab_size': 5000,        # 字典大小
    'l2_reg_lambda': 0.0,      # l2正则化参数
    'device': '/cpu:0',        # 设置device
    'batch_size': 100,         # batch大小
    'num_epochs': 2,           # epoch数目
    'evaluate_every': 100,     # 每隔多少步打印一次验证集结果
    'checkpoint_every': 100,   # 每隔多少步保存一次模型
    'num_checkpoints': 5,      # 最多保存模型的个数
    'allow_soft_placement': True,   # 是否允许程序自动选择备用device
    'log_device_placement': False,  # 是否允许在终端打印日志文件
    'train_test_dev_rate': [0.97, 0.02, 0.01],   # 训练集，测试集，验证集比例
    'data_path': './data/cnews.test.txt',    # 数据路径  格式：标签\t文本
    'learning_rate': 0.003,             # 学习率
    'dropout_keep_prob': 0.5,           # 训练过程中dropout大小
    'vocab_path': './vocabs',           # 保存词典路径
    'hidden_size': 128
}