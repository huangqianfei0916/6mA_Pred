
### Data preprocessing
* The first step：
```
python fasta2word.py -fasta xxx.fasta
```
* The second step(XXX is the first step to get the word segmentation file, which will eventually generate a word vector model of word2vec.model)
```
python word2vec.py -word xxx
```


### Parameter Settings
*********
|parameter|values|
|:-|:-|  
|-train_data_path|train file|    
-train_pos|    	Number of positive examples  
-train_neg|     Number of negative examples  
-seed     |    seed
-freeze     |     Embedding freeze
-embedding1      |  Word embedding model
-batch_size|    batch size
-test_data_path  | 	  test file
-test_pos   | 	Number of positive examples   
-test_neg  |		Number of negative examples
-rnn_layers|lstm layers
-fix_len   |  fix length
-learning_rate   | learning rate
-dropout| dropout
-hidden_dims|hidden_dim of lstm
-num_epochs|epochs
-init|default：True
************
### Example
* Independent testing（When no test set is provided, the training set is partitioned）
```py
python start_train.py
-train_data_path
train_data.txt
-train_pos
154000
-train_neg
154000
-embedding1
word2vec.model
-fix_len
39
# Nonessential parameter
-test_data_path
test_data.txt
-test_pos
880
-test_neg
880
```
