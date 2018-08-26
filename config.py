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

# DataSet
data_arg = add_arg_group('DataSet')

# Train/Test hyper-parameters
train_arg = add_arg_group('Training')

# Misc
misc_arg = add_arg_group('Misc')
