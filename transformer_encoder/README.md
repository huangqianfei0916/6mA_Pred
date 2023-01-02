

### 数据预处理
* 对fasta文件进行切词，模型采用3mer
```
python fasta2word.py -fasta input.fasta -kmer 3
```

### 运行,需要提供自己的词典-word.dict
fix_len:决定pos emd的长度
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
