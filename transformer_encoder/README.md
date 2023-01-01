
### 数据预处理
* 第一步：
```
python fasta2word.py -fasta input.fasta
```
* 第二步(word.txt是第一步中得到分词文件，最终会生成一个word2vec.model的词向量model)
```
python word2vec.py -word word.txt
```

### 运行方式
****************
```
python train.py
-dict
word.dict
-train_data_path
word.txt
-train_pos
880
-train_neg
880
-fix_len
39
# 非必须参数
-test_data_path
test_data.txt
-test_pos
880
-test_neg
880
```
