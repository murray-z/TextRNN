# 基于字向量TextCNN文本分类

# 目录说明
<pre>
-data    存放数据(样例见./data/cnews.txt, https://pan.baidu.com/s/1aPkGkfTgsz6IlTLzfXkJ1w)
-vocabs  存放字典文件，训练过程中自动生成
-runs    存放模型及tensorboard文件，训练过程中自动生成
-data_helper.py   处理数据
-config.py        配置文件
-text_rnn.py      TextRNN类
-train.py         训练模型
-predict.py       预测
</pre>

# 训练
```
python train.py
```

# 预测
```
python predict.py
```
