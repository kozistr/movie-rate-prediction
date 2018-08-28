import gc
import time
import h5py
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from model import charcnn
from config import get_config
from matplotlib import pyplot as plt
from dataloader import Doc2VecEmbeddings, DataLoader, DataIterator


parser = argparse.ArgumentParser(description='train/test movie review classification model')
parser.add_argument('--checkpoint', type=str, help='pre-trained model', default=None)
parser.add_argument('--save_to_h5', type=str, help='saving vectorized processed data into h5', default=None)
parser.add_argument('--load_from_h5', type=str, help='loading vectorized processed data from h5', default=None)
args = parser.parse_args()

# parsed args
checkpoint = args.checkpoint
save_to_h5 = args.save_to_h5
load_from_h5 = args.load_from_h5

# Configuration
config, _ = get_config()

np.random.seed(config.seed)
tf.set_random_seed(config.seed)


# hand-made samples
# you can replace this part to your custom DataSet :)
samples = [
    {'rate': 9, 'comment': "대박 개쩔어요!!"},
    {'rate': 7, 'comment': "띵작 그런데 좀 아쉽다..."},
    {'rate': 2, 'comment': "쓰레기... 에반데"},
]


def data_distribution(y_: np.array, size: int = 10, img: str = 'dist.png') -> np.array:
    # showing data distribution
    y_dist = np.zeros((size,), dtype=np.int32)
    for y in tqdm(y_):
        y_dist[np.argmax(y, axis=-1)] += 1

    plt.figure(figsize=(10, 8))

    plt.xlabel('rate')
    plt.ylabel('frequency')
    plt.grid(True)

    plt.bar(range(size), y_dist, width=.35, align='center', alpha=.5, label='rainfall')
    plt.xticks(range(size), list(range(1, size + 1)))

    plt.savefig(img)
    plt.show()

    return y_dist


if __name__ == '__main__':
    # Loading Doc2Vec Model
    if config.use_pre_trained_embeds:
        # Doc2Vec Loader
        vec = Doc2VecEmbeddings(config.d2v_model, config.embed_size)
        if config.verbose:
            print("[+] Doc2Vec loaded! Total %d pre-trained sentences, %d dims" % (len(vec), config.embed_size))
    else:
        raise NotImplementedError("[-] character-level pre-processing not yet ready :(")

    if not load_from_h5:
        # DataSet Loader
        ds = DataLoader(file=config.processed_dataset,
                        n_classes=config.n_classes,
                        analyzer=None,
                        is_analyzed=True,
                        use_save=False,
                        config=config)
        ds_len = len(ds)

        if config.verbose:
            print("[+] DataSet loaded! Total %d samples" % ds_len)

        # words Vectorization # type conversion
        x_data = np.zeros((ds_len, config.embed_size), dtype=np.float32)
        y_data = np.zeros((ds_len, config.n_classes), dtype=np.uint8)

        for idx in tqdm(range(ds_len)):
            x_data[idx] = vec.sent_to_vec(ds.sentences[idx])
            y_data[idx] = np.asarray(ds.labels[idx])
            ds.sentences[idx] = None
            ds.labels[idx] = None

        if config.verbose:
            print("[+] conversion finish! x_data, y_data loaded!")

        # delete DataSetLoader() from memory
        ds = None
        gc.collect()

        if save_to_h5:
            print("[*] start writing .h5 file...")
            with h5py.File(save_to_h5, 'w') as h5fs:
                h5fs.create_dataset('comment', data=np.array(x_data))
                h5fs.create_dataset('rate', data=np.array(y_data))

            if config.verbose:
                print("[+] data saved into h5 file!")
    else:
        with h5py.File(load_from_h5, 'r') as f:
            x_data = np.array(f['comment'], dtype=np.float32)
            y_data = np.array(f['rate'], dtype=np.uint8)

            if config.verbose:
                print("[+] data loaded from h5 file!")
                print("[*] comment : ", x_data.shape)
                print("[*] rate    : ", y_data.shape)

    # show data rate distribution
    y_dist = data_distribution(y_data)

    # resizing the amount of rate-10 data
    # 2.5M to 500K # downsize to 20%
    rate_10_idx = [idx for idx, y in tqdm(enumerate(y_data)) if np.argmax(y, axis=-1) == 9]
    rand_idx = np.random.choice(rate_10_idx, len(rate_10_idx) // 5)

    x_data = np.delete(x_data, rand_idx).reshape(-1, config.embed_size)
    y_data = np.delete(y_data, rand_idx).reshape(-1, config.n_classes)

    if config.verbose:
        print("[*] refined comment : ", x_data.shape)
        print("[*] refined rate    : ", y_data.shape)

    data_size = x_data.shape[0]

    # DataSet Iterator
    di = DataIterator(x=x_data, y=y_data, batch_size=config.batch_size)

    if config.is_train:
        # GPU configure
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        with tf.Session(config=gpu_config) as s:
            if config.model == 'charcnn':
                # Model Loaded
                model = charcnn.CharCNN(s=s,
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

            restored_epochs = global_step // (data_size // config.batch_size)
            for epoch in range(restored_epochs, config.epochs):
                for x_train, y_train in di.iterate():
                    # training
                    _, loss = s.run([model.opt, model.loss],
                                    feed_dict={
                                        model.x: x_train,
                                        model.y: y_train,
                                        model.do_rate: config.drop_out,
                                    })

                    if global_step % config.logging_step == 0:
                        print("[*] epoch %d global step %d" % (epoch, global_step), " loss : {:.8f}".format(loss))

                        # predictions
                        for sample_data in samples:
                            sample = vec.sent_to_vec(sample_data['comment']).reshape(-1, config.embed_size)
                            predict = s.run(model.prediction,
                                            feed_dict={
                                                model.x: sample,
                                                model.do_rate: .0,
                                            })
                            print("[*] predict %030s : %d (expected %d)" % (sample_data['comment'],
                                                                            predict + 1,
                                                                            sample_data['rate']))

                        # summary
                        summary = s.run(model.merged,
                                        feed_dict={
                                            model.x: x_train,
                                            model.y: y_train,
                                            model.do_rate: .0,
                                        })

                        # Summary saver
                        model.writer.add_summary(summary, global_step)

                        # Model save
                        model.saver.save(s, config.pretrained + '%s.ckpt' % config.model, global_step=global_step)

                    global_step += 1

            end_time = time.time()

            print("[+] Training Done! Elapsed {:.8f}s".format(end_time - start_time))
    else:  # Test
        pass
