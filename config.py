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

# DataSet

# Train/Test hyper-parameters

# Misc
