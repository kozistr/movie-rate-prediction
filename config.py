import argparse


args_list = []
parser = argparse.ArgumentParser()


def add_arg_group(name: str):
    arg = parser.add_argument_group(name)
    args_list.append(arg)
    return arg


def get_config():
    cfg, un_parsed = parser.parse_known_args()
    return cfg, un_parsed


# Network
network_arg = add_arg_group('Network')
network_arg.add_argument('--mode', type=str, default='static', choices=['static', 'non-static'])
network_arg.add_argument('--model', type=str, default='charcnn', choices=['charcnn', 'charrnn'])
network_arg.add_argument('--n_classes', type=int, default=1)
network_arg.add_argument('--use_pre_trained_embeds', type=bool, default=True,
                         help='using Doc2Vec as embedding. If False, use character-level embedding')
network_arg.add_argument('--kernel_size', type=list, default=[2, 3, 4, 5],
                         help='conv1d kernel size')
network_arg.add_argument('--fc_unit', type=int, default=1024)
network_arg.add_argument('--drop_out', type=int, default=.2,
                         help='dropout rate')
network_arg.add_argument('--use_leaky_relu', type=bool, default=False)
network_arg.add_argument('--act_threshold', type=float, default=1e-6,
                         help='used at ThresholdReLU')

# DataSet
data_arg = add_arg_group('DataSet')
data_arg.add_argument('--embed_size', type=int, default=300,
                      help='the size of Doc2Vec embedding vector')
data_arg.add_argument('--batch_size', type=int, default=256)
data_arg.add_argument('--n_threads', type=int, default=8,
                      help='the number of workers for speeding up')

# Train/Test hyper-parameters
train_arg = add_arg_group('Training')
train_arg.add_argument('--is_train', type=bool, default=True)
train_arg.add_argument('--epochs', type=int, default=100)
train_arg.add_argument('--logging_step', type=int, default=500)
train_arg.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
train_arg.add_argument('--lr', type=float, default=2e-4)
train_arg.add_argument('--lr_decay', type=float, default=.95)
train_arg.add_argument('--lr_lower_boundary', type=float, default=2e-5)
train_arg.add_argument('--test_size', type=float, default=.2)

# Korean words Pre-Processing
nlp_model = add_arg_group('NLP')
nlp_model.add_argument('--w2v_model', type=str, default='./w2v/ko_w2v.model')
nlp_model.add_argument('--d2v_model', type=str, default='./w2v/ko_d2v.model')
nlp_model.add_argument('--analyzer', type=str, default='mecab', choices=['mecab', 'hannanum', 'twitter'],
                       help='korean pos analyzer')
nlp_model.add_argument('--use_correct_spacing', type=bool, default=False,
                       help='resolving sentence spacing problem but taking lots of time...')
nlp_model.add_argument('--use_normalize', type=bool, default=True)
nlp_model.add_argument('--vec_lr', type=float, default=2.5e-2)
nlp_model.add_argument('--vec_min_lr', type=float, default=2.5e-2)
nlp_model.add_argument('--vec_lr_decay', type=float, default=2e-3)

# Misc
misc_arg = add_arg_group('Misc')
misc_arg.add_argument('--query_path', type=str, default='./comments/')
misc_arg.add_argument('--dataset', type=str, default='data.csv')
misc_arg.add_argument('--processed_dataset', type=str, default='tagged_data.csv',
                      help='already processed data file')
misc_arg.add_argument('--pretrained', type=str, default='./ml_model/')
misc_arg.add_argument('--w2v_model', type=str, default='ko_w2v.model')
misc_arg.add_argument('--d2v_model', type=str, default='ko_d2v.model')
misc_arg.add_argument('--seed', type=int, default=1337)
misc_arg.add_argument('--jvm_path', type=str, default="C:\\Program Files\\Java\\jre-9\\bin\\server\\jvm.dll")
misc_arg.add_argument('--verbose', type=bool, default=True)

# DB
db_arg = add_arg_group('DB')
db_arg.add_argument('--host', type=str, default='localhost')
db_arg.add_argument('--user', type=str, default='root')
db_arg.add_argument('--password', type=str, default='1111')
db_arg.add_argument('--db', type=str, default='movie')
db_arg.add_argument('--charset', type=str, default='utf8')
