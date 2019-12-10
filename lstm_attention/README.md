### 参数设置
*********
|参数|取值|
|:-|:-|  
|-train_data_path|训练集分词文件|    
-train_pos|    	正例数  
-train_neg|       	反例数  
-seed     |    随即种子
-freeze     |     是否更新词向量
-embedding1      |  词向量模型1
-embedding2|   	 词向量模型2 
-batch_size|    批量大小 
-test_data_path  | 	  测试集分词文件
-test_pos   | 	正例数  
-test_neg  |		反例数  
-rnn_layers|lstm层数
-fix_len   |		补全长度
-learning_rate   |学习率
-dropout| 防止过拟合
-hidden_dims|lstm的hidden_dim
-num_epochs|迭代次数
-init|是否使用初始化的词向量；默认True
************
### Example
```py
python start_train.py
-train_data_path
C:\\Users\\hqf\\PycharmProjects\\Bio/attention/lstm_attention/data/train_data.txt
-train_pos
154000
-train_neg
154000
-embedding1
model.model
-fix_len
39
-test_data_path
C:\\Users\\hqf\\PycharmProjects\\Bio/attention/lstm_attention/data/test_data.txt
-test_pos
880
-test_neg
880
```
