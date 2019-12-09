import torch


class DefaultConfig(object):

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model_path
    model1 = '..\\data\\w2v\\model.model'

    #data
    train_data_path = "..\\data\\train_data.txt"
    test_data_path = "..\\data\\test_data.txt"

    train_pos = 154000
    train_neg = 154000

    test_pos = 880
    test_neg = 880
    # 若是等长就是序列的单词数，若是不等长，就是 截断或者补全的长度
    fix_len = 39

    """模型结构参数"""
    model_name = 'lstm-attention'
    freeze = True
    num_classes = 2
    static=True
    hidden_dims=100
    rnn_layers=2

    """训练参数"""
    random_seed = 2019
    use_gpu = True
    learning_rate = 0.01
    weight_decay = 0
    num_epochs = 200
    data_shuffle = True
    batch_size = 64

    '''transformer'''
    src_vocab_size=64
    max_token_seq_len=39
    d_k =64
    d_v = 64
    d_model = 100
    d_word_vec =100
    d_inner = 512
    n_layers = 1
    n_head = 8
    dropout = 0.1
