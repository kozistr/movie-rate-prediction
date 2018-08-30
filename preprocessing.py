import logging
import argparse

from tqdm import tqdm
from config import get_config
from dataloader import DataLoader
from collections import namedtuple


# Argument parser
parser = argparse.ArgumentParser(description='Pre-Processing NAVER Movie Review Comment')
parser.add_argument('--load_from', type=str, help='load DataSet from db or csv', default='db', choices=['db', 'csv'])
parser.add_argument('--vector', type=str, help='d2v or w2v', choices=['d2v', 'w2v'], default='w2v')
parser.add_argument('--is_analyzed', type=bool, help='already analyzed data', default=False)
args = parser.parse_args()

config, _ = get_config()  # global configuration

vec = args.vector
load_from = args.load_from
is_analyzed = args.is_analyzed


def w2v_training(data: list) -> bool:
    from gensim.models import word2vec

    global config

    # flatten & remove duplicates
    # data = list(set(sum(data, [])))

    # word2vec Training
    w2v_config = {
        'sentences': data,
        'batch_words': 12800,
        'size': config.embed_size,
        'window': 5,
        'min_count': 1,
        'negative': 5,
        'alpha': config.vec_lr,
        'sg': 1,
        'iter': 10,
        'seed': config.seed,
        'workers': config.n_threads,
    }
    w2v_model = word2vec.Word2Vec(**w2v_config)
    w2v_model.wv.init_sims(replace=True)

    w2v_model.save(config.w2v_model)
    return True


def d2v_training(sentences: list, rates: list, epochs=10) -> bool:
    from gensim.models import Doc2Vec

    global config

    # data processing to fit in Doc2Vec
    taggedDocs = namedtuple('TaggedDocument', 'words tags')
    tagged_data = [taggedDocs(s, r) for s, r in zip(sentences, rates)]

    d2v_config = {
        'dm': 1,
        'dm_concat': 1,
        'vector_size': config.embed_size,
        'negative': 3,
        'hs': 0,
        'alpha': config.vec_lr,
        'min_alpha': config.vec_min_lr,
        'min_count': 3,
        'window': 5,
        'seed': config.seed,
        'workers': config.n_threads,
    }
    d2v_model = Doc2Vec(**d2v_config)
    d2v_model.build_vocab(tagged_data)

    total_examples = len(sentences)
    for _ in tqdm(range(epochs)):
        # Doc2Vec training
        d2v_model.train(tagged_data, total_examples=total_examples, epochs=d2v_model.iter)

        # LR Scheduler
        d2v_model.alpha -= config.vec_lr_decay
        d2v_model.min_alpha = d2v_model.alpha

    d2v_model.save(config.d2v_model)
    return True


def main():
    # Data Loader
    if is_analyzed:
        data_loader = DataLoader(file=config.processed_dataset,
                                 is_analyzed=True,
                                 use_save=False,
                                 config=config)  # processed data
    else:
        data_loader = DataLoader(file=config.dataset,
                                 load_from=load_from,
                                 use_save=True,
                                 fn_to_save=config.processed_dataset,
                                 config=config)  # not processed data

    x_data, y_data = data_loader.sentences, data_loader.labels

    # logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if vec == 'd2v':
        d2v_training(x_data, y_data)  # d2v Training
    elif vec == 'w2v':
        w2v_training(x_data)          # w2v Training


if __name__ == "__main__":
    main()
