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
network_arg.add_argument('--kernel_size', type=list, default=[1, 2, 3, 4],
                         help='conv1d kernel size')
network_arg.add_argument('--fc_unit', type=int, default=1024)
network_arg.add_argument('--drop_out', type=int, default=.8,
                         help='dropout rate')

# DataSet
data_arg = add_arg_group('DataSet')
data_arg.add_argument('--embed_size', type=int, default=300,
                      help='the size of Doc2Vec embedding vector')
data_arg.add_argument('--batch_size', type=int, default=128)
data_arg.add_argument('--n_threads', type=int, default=8,
                      help='the number of workers for speeding up')

# Train/Test hyper-parameters
train_arg = add_arg_group('Training')
train_arg.add_argument('--is_train', type=bool, default=True)
train_arg.add_argument('--max_step', type=int, default=1e6)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--lr', type=float, default=8e-4)

# Misc
misc_arg = add_arg_group('Misc')
