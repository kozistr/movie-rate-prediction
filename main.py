import time
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from model import charcnn
from config import get_config
from sklearn.model_selection import train_test_split
from dataloader import Word2VecEmbeddings, Doc2VecEmbeddings, DataLoader, DataIterator


parser = argparse.ArgumentParser(description='train/test movie review classification model')
parser.add_argument('--checkpoint', type=str, help='pre-trained model', default=None)
parser.add_argument('--save_to_h5', type=str, help='saving vectorized processed data into h5', default=None)
parser.add_argument('--load_from_h5', type=str, help='loading vectorized processed data from h5', default=None)
parser.add_argument('--refine_data', type=bool, help='solving data imbalance problem', default=False)
args = parser.parse_args()

# parsed args
checkpoint = args.checkpoint
save_to_h5 = args.save_to_h5
refine_data = args.refine_data
load_from_h5 = args.load_from_h5

# Configuration
config, _ = get_config()

np.random.seed(config.seed)
tf.set_random_seed(config.seed)


# hand-made samples
# you can replace this part to your custom DataSet :)
samples = [
    {'rate': 10, 'comment': "이건 10점 안줄 수 가 없다. 닥추"},
    {'rate': 9, 'comment': "대박 개쩔어요!!"},
    {'rate': 7, 'comment': "띵작 그런데 좀 아쉽다..."},
    {'rate': 5, 'comment': "그냥 그럼"},
    {'rate': 2, 'comment': "쓰레기... 에반데"},
]


def data_distribution(y_: np.array, size: int = 10, img: str = 'dist.png') -> np.array:
    from matplotlib import pyplot as plt

    # showing data distribution
    y_dist = np.zeros((10,), dtype=np.int32)
    for y in tqdm(y_):
        if size == 1:
            y_dist[y - 1] += 1
        else:
            y_dist[np.argmax(y, axis=-1)] += 1

    plt.figure(figsize=(10, 8))

    plt.xlabel('rate')
    plt.ylabel('frequency')
    plt.grid(True)

    plt.bar(range(size), y_dist, width=.35, align='center', alpha=.5, label='rainfall')
    plt.xticks(range(10), list(range(1, 11)))

    plt.savefig(img)
    plt.show()

    return y_dist


def load_trained_embeds(embed_mode: str = 'w2v'):
    if embed_mode == 'd2v':
        vec = Doc2VecEmbeddings(config.d2v_model, config.embed_size)  # Doc2Vec Loader
        if config.verbose:
            print("[+] Doc2Vec loaded! Total %d pre-trained sentences, %d dims" % (len(vec), config.embed_size))
    elif embed_mode == 'd2v':
        vec = Word2VecEmbeddings(config.w2v_model, config.embed_size)  # WOrd2Vec Loader
        if config.verbose:
            print("[+] Word2Vec loaded! Total %d pre-trained words, %d dims" % (len(vec), config.embed_size))
    else:
        raise NotImplementedError("[-] character-level pre-processing not yet ready :(")
    return vec


if __name__ == '__main__':
    # Stage 1 : loading trained embeddings
    vectors = load_trained_embeds(config.use_pre_trained_embeds)

    # Stage 2 : loading DataSet



    # show data rate distribution
    # y_dist = data_distribution(y_data, config.n_classes)

    if refine_data:
        # resizing the amount of rate-10 data
        # 2.5M to 500K # downsize to 20%
        if not config.n_classes == 1:
            rate_10_idx = [idx for idx, y in tqdm(enumerate(y_data)) if np.argmax(y, axis=-1) == 9]
        else:
            rate_10_idx = [idx for idx, y in tqdm(enumerate(y_data)) if y == 10]
        rand_idx = np.random.choice(rate_10_idx, 4 * len(rate_10_idx) // 5)

        x_data = np.delete(x_data, rand_idx, axis=0).reshape(-1, config.embed_size)
        y_data = np.delete(y_data, rand_idx, axis=0).reshape(-1, config.n_classes)

        if config.verbose:
            print("[*] refined comment : ", x_data.shape)
            print("[*] refined rate    : ", y_data.shape)

    # shuffle/split data
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, random_state=config.seed,
                                                          test_size=config.test_size, shuffle=True)
    if config.verbose:
        print("[*] train/test (%.1f/%.1f) split!" % (1. - config.test_size, config.test_size))

    data_size = x_data.shape[0]

    # DataSet Iterator
    di = DataIterator(x=x_train, y=y_train, batch_size=config.batch_size)

    if config.is_train:
        # GPU configure
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        with tf.Session(config=gpu_config) as s:
            if config.model == 'charcnn':
                # Model Loaded
                model = charcnn.CharCNN(s=s,
                                        mode=config.mode,
                                        n_classes=config.n_classes,
                                        optimizer=config.optimizer,
                                        dims=config.embed_size,
                                        lr=config.lr,
                                        lr_decay=config.lr_decay,
                                        lr_lower_boundary=config.lr_lower_boundary,
                                        th=config.act_threshold,
                                        summary=config.pretrained)
            elif config.model == 'charrnn':
                raise NotImplementedError("[-] Not Implemented Yet")
            else:
                raise NotImplementedError("[-] Not Implemented Yet")

            if config.verbose:
                print("[+] %s model loaded" % config.model)

            # Initializing
            s.run(tf.global_variables_initializer())

            global_step = 0
            if checkpoint:
                print("[*] Reading checkpoints...")

                ckpt = tf.train.get_checkpoint_state(config.pretrained)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    model.saver.restore(s, ckpt.model_checkpoint_path)

                    global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                    print("[+] global step : %d" % global_step, " successfully loaded")
                else:
                    print('[-] No checkpoint file found')

            start_time = time.time()

            best_loss = 100.  # initial value
            restored_epochs = global_step // (data_size // config.batch_size)
            for epoch in range(restored_epochs, config.epochs):
                for x_tr, y_tr in di.iterate():
                    # training
                    _, loss, acc = s.run([model.opt, model.loss, model.accuracy],
                                         feed_dict={
                                             model.x: x_tr,
                                             model.y: y_tr,
                                             model.do_rate: config.drop_out,
                                         })

                    if global_step % config.logging_step == 0:
                        summary, valid_loss, valid_acc = s.run([model.merged, model.loss, model.accuracy],
                                                               feed_dict={
                                                                   model.x: x_valid,
                                                                   model.y: y_valid,
                                                                   model.do_rate: .0,
                                                               })
                        print("[*] epoch %03d global step %07d" % (epoch, global_step),
                              " train_loss : {:.8f} train_acc : {:.4f}".format(loss, acc),
                              " valid_loss : {:.8f} valid_acc : {:.4f}".format(valid_loss, valid_acc))

                        # Summary saver
                        model.writer.add_summary(summary, global_step)

                        # Model save
                        model.saver.save(s, config.pretrained + '%s.ckpt' % config.model,
                                         global_step=global_step)

                        if valid_loss < best_loss:
                            print("[+] model improved {:.6f} to {:.6f}".format(best_loss, loss))
                            model.best_saver.save(s, config.pretrained + '%s.ckpt' % config.model,
                                                  global_step=global_step)
                        print()

                    global_step += 1

                # predictions
                for sample_data in samples:
                    sample = vec.sent_to_vec(sample_data['comment']).reshape(-1, config.embed_size)
                    predict = s.run(model.prediction,
                                    feed_dict={
                                        model.x: sample,
                                        model.do_rate: .0,
                                    })
                    print("[*] predict %050s : %d (expected %d)" % (sample_data['comment'],
                                                                    predict + 1,
                                                                    sample_data['rate']))

            end_time = time.time()

            print("[+] Training Done! Elapsed {:.8f}s".format(end_time - start_time))
    else:  # Test
        pass
