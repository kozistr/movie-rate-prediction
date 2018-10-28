import time
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from config import get_config
from model.textcnn import TextCNN
from model.textrnn import TextRNN
from sklearn.model_selection import train_test_split
from dataloader import Word2VecEmbeddings, Doc2VecEmbeddings, CharVecEmbeddings, DataLoader, DataIterator


parser = argparse.ArgumentParser(description='train/test movie review classification model')
parser.add_argument('--checkpoint', type=str, help='pre-trained model', default=None)
parser.add_argument('--refine_data', type=bool, help='solving data imbalance problem', default=False)
args = parser.parse_args()

# parsed args
checkpoint = args.checkpoint
refine_data = args.refine_data

# Configuration
config, _ = get_config()

np.random.seed(config.seed)
tf.set_random_seed(config.seed)


def data_distribution(y_, size=10, img='dist.png'):
    """
    movie rate data distribution via plot chart
    :param y_: rate data, numpy array
    :param size: classes, int
    :param img: save to, str
    :return: numpy array
    """
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


def load_trained_embeds(embed_mode='char'):
    """
    :param embed_mode: embedding mode, str
    :return: embedding vector, numpy array
    """
    if embed_mode == 'd2v':
        vec = Doc2VecEmbeddings(config.d2v_model, config.embed_size)  # Doc2Vec Loader
        if config.verbose:
            print("[+] Doc2Vec loaded! Total %d pre-trained sentences, %d dims" % (len(vec), config.embed_size))
    elif embed_mode == 'w2v':
        vec = Word2VecEmbeddings(config.w2v_model, config.embed_size)  # WOrd2Vec Loader
        if config.verbose:
            print("[+] Word2Vec loaded! Total %d pre-trained words, %d dims" % (len(vec), config.embed_size))
    else:
        vec = CharVecEmbeddings(config.sequence_length)
        if config.verbose:
            print("[+] Using Char2Vec, %d dims" % config.embed_size)
    return vec


if __name__ == '__main__':
    embed_type = config.use_pre_trained_embeds

    # Stage 1 : loading trained embeddings
    vectors = load_trained_embeds(embed_type)

    # Stage 2 : loading tokenize data
    if config.use_pre_trained_embeds == 'c2v':  # Char2Vec
        ds = DataLoader(file=config.processed_dataset,  # None
                        fn_to_save=None,  # config.processed_dataset,
                        load_from='db',
                        n_classes=config.n_classes,
                        analyzer='char',
                        is_analyzed=True,  # False,
                        use_save=False,  # True,
                        config=config)  # DataSet Loader

        ds_len = len(ds)

        x_data = np.zeros((ds_len, config.sequence_length), dtype=np.int32)
        for i in tqdm(range(ds_len)):
            sent = [x.replace('', ' ') for x in ds.sentences[i][:config.sequence_length] if x == '']
            sent = vectors.decompose_str_as_one_hot(' '.join(sent).strip('\n'), warning=False)
            x_data[i] = np.pad(sent,
                               (0, config.sequence_length - len(sent)), 'constant', constant_values=0)
    else:  # Word2Vec / Doc2Vec
        ds = DataLoader(file=config.processed_dataset,
                        n_classes=config.n_classes,
                        analyzer=None,
                        is_analyzed=True,
                        use_save=False,
                        config=config)  # DataSet Loader

        ds_len = len(ds)

        x_data = np.zeros((ds_len, config.sequence_length), dtype=np.int32)
        for i in tqdm(range(ds_len)):
            sent = ds.sentences[i][:config.sequence_length]
            x_data[i] = np.pad(vectors.words_to_index(sent),
                               (0, config.sequence_length - len(sent)), 'constant', constant_values=config.vocab_size)

    x_data = np.array(x_data)
    y_data = np.array(ds.labels).reshape(-1, config.n_classes)

    ds = None
    del ds

    if config.verbose:
        print("[*] sentence to w2v index conversion finish!")

    if refine_data:
        # resizing the amount of rate-10 data
        # 2.5M to 500K # downsize to 20%
        if not config.n_classes == 1:
            rate_10_idx = [idx for idx, y in tqdm(enumerate(y_data)) if np.argmax(y, axis=-1) == 9]
        else:
            rate_10_idx = [idx for idx, y in tqdm(enumerate(y_data)) if y == 10]

        rand_idx = np.random.choice(rate_10_idx, 4 * len(rate_10_idx) // 5)

        x_data = np.delete(x_data, rand_idx, axis=0).reshape(-1, config.sequence_length)
        y_data = np.delete(y_data, rand_idx, axis=0).reshape(-1, config.n_classes)

        if config.verbose:
            print("[*] refined comment : ", x_data.shape)
            print("[*] refined rate    : ", y_data.shape)

    # shuffle/split data
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, random_state=config.seed,
                                                          test_size=config.test_size, shuffle=True)
    if config.verbose:
        print("[*] train/test %d/%d(%.1f/%.1f) split!" % (len(y_train), len(y_valid),
                                                          1. - config.test_size, config.test_size))

    del x_data, y_data

    data_size = x_train.shape[0]

    # DataSet Iterator
    di = DataIterator(x=x_train, y=y_train, batch_size=config.batch_size)

    if config.device == 'gpu':
        dev_config = tf.ConfigProto()
        dev_config.gpu_options.allow_growth = True
    else:
        dev_config = None

    if config.is_train:
        with tf.Session(config=dev_config) as s:
            if config.model == 'charcnn':
                # Model Loaded
                model = TextCNN(s=s,
                                mode=config.mode,
                                w2v_embeds=vectors.embeds if not embed_type == 'c2v' else None,
                                n_classes=config.n_classes,
                                optimizer=config.optimizer,
                                kernel_sizes=config.kernel_size,
                                n_filters=config.filter_size,
                                n_dims=config.embed_size,
                                vocab_size=config.character_size if embed_type == 'c2v' else config.vocab_size + 1,
                                sequence_length=config.sequence_length,
                                lr=config.lr,
                                lr_decay=config.lr_decay,
                                lr_lower_boundary=config.lr_lower_boundary,
                                fc_unit=config.fc_unit,
                                th=config.act_threshold,
                                grad_clip=config.grad_clip,
                                summary=config.pretrained)
            elif config.model == 'charrnn':
                model = TextRNN(s=s,
                                mode=config.mode,
                                w2v_embeds=vectors.embeds if not embed_type == 'c2v' else None,
                                n_classes=config.n_classes,
                                optimizer=config.optimizer,
                                n_gru_cells=config.n_gru_cells,
                                n_gru_layers=config.n_gru_layers,
                                n_attention_size=config.n_attention_size,
                                n_dims=config.embed_size,
                                vocab_size=config.character_size if embed_type == 'c2v' else config.vocab_size + 1,
                                sequence_length=config.sequence_length,
                                lr=config.lr,
                                lr_decay=config.lr_decay,
                                lr_lower_boundary=config.lr_lower_boundary,
                                fc_unit=config.fc_unit,
                                grad_clip=config.grad_clip,
                                summary=config.pretrained)
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
            batch_size = config.batch_size
            model.global_step.assign(tf.constant(global_step))
            restored_epochs = global_step // (data_size // batch_size)
            for epoch in range(restored_epochs, config.epochs):
                for x_tr, y_tr in di.iterate():
                    # training
                    _, loss, acc = s.run([model.train_op, model.loss, model.accuracy],
                                         feed_dict={
                                             model.x: x_tr,
                                             model.y: y_tr,
                                             model.do_rate: config.drop_out,
                                         })

                    if global_step and global_step % config.logging_step == 0:
                        # validation
                        rand_idx = np.random.choice(np.arange(len(y_valid)), len(y_valid) // 50)  # 2% of valid data

                        x_va, y_va = x_valid[rand_idx], y_valid[rand_idx]

                        valid_loss, valid_acc = .0, .0
                        valid_iter = len(y_va) // batch_size
                        for i in tqdm(range(0, valid_iter)):
                            v_loss, v_acc = s.run([model.loss, model.accuracy],
                                                  feed_dict={
                                                      model.x: x_va[batch_size * i:batch_size * (i + 1)],
                                                      model.y: y_va[batch_size * i:batch_size * (i + 1)],
                                                      model.do_rate: .0,
                                                  })
                            valid_loss += v_loss
                            valid_acc += v_acc

                        valid_loss /= valid_iter
                        valid_acc /= valid_iter

                        print("[*] epoch %03d global step %07d" % (epoch, global_step),
                              " train_loss : {:.8f} train_acc : {:.4f}".format(loss, acc),
                              " valid_loss : {:.8f} valid_acc : {:.4f}".format(valid_loss, valid_acc))

                        # summary
                        summary = s.run(model.merged,
                                        feed_dict={
                                            model.x: x_tr,
                                            model.y: y_tr,
                                            model.do_rate: .0,
                                        })

                        # Summary saver
                        model.writer.add_summary(summary, global_step)

                        # Model save
                        model.saver.save(s, config.pretrained + '%s.ckpt' % config.model,
                                         global_step=global_step)

                        if valid_loss < best_loss:
                            print("[+] model improved {:.6f} to {:.6f}".format(best_loss, valid_loss))
                            best_loss = valid_loss

                            model.best_saver.save(s, config.pretrained + '%s-best_loss.ckpt' % config.model,
                                                  global_step=global_step)
                        print()

                    model.global_step.assign_add(tf.constant(1))
                    global_step += 1

            end_time = time.time()

            print("[+] Training Done! Elapsed {:.8f}s".format(end_time - start_time))
    else:  # test
        pass
