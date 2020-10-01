
### 数据预处理
* 第一步：
```
python fasta2word.py -fasta xxx.fasta
```
* 第二步(xxx是第一步中得到分词文件，最终会生成一个word2vec.model的词向量model)
```
python word2vec.py -word xxx
```

### 运行方式
****************
* 修改 config.py文件中参数配置即可。
* model1：训练好的词向量模型，样例参考 w2v/model.model（这里是按照gensim读取word2vec词向量模型的方式读取的）
* train_data_path : 训练集分词文件
* test_data_path ： 测试集分词文件
* train_pos ：训练集正样本个数
* train_neg ：训练集负样本个数
* test_pos ：测试集正样本个数
* test_neg ：测试集负样本个数
* fix_len ： 一条记录词的个数（或者序列不等长时代表截断的长度）
* src_vocab_size： 词的个数
* max_token_seq_len：同fix_len
************
### 调节参数
*************
* num_epochs ：迭代次数
* learning_rate： 学习率
* dropout： 防止过拟合
* n_head：多头注意力的多头数
* n_layers：encoder的层数
