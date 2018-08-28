import time
import argparse
import numpy as np
import tensorflow as tf

from model import charcnn
from config import get_config
from dataloader import Doc2VecEmbeddings, DataLoader, DataIterator


parser = argparse.ArgumentParser(description='train/test movie review classification model')
parser.add_argument('--checkpoint', type=str, help='pre-trained model', default=None)
args = parser.parse_args()

# parsed args
checkpoint = args.checkpoint

# Configuration
config, _ = get_config()

np.random.seed(config.seed)
tf.set_random_seed(config.seed)


if __name__ == '__main__':
    # DataSet Loader
    ds = DataLoader(file=config.processed_dataset,
                    n_classes=config.n_classes,
                    analyzer=None,
                    is_analyzed=True,
                    use_save=False,
                    config=config)

    if config.verbose:
        print("[+] DataSet loaded! Total %d samples" % len(ds.labels))

    # DataSet Iterator
    di = DataIterator(x=ds.sentences, y=ds.labels, batch_size=config.batch_size)

    # Doc2Vec Loader
    vec = Doc2VecEmbeddings(config.d2v_model, config.embed_size)
    if config.verbose:
        print("[+] Doc2Vec loaded! Total %d pre-trained sentences, %d dims" % (len(vec), config.n_dims))

    if config.is_train:
        # GPU configure
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as s:
            if config.model == 'charcnn':
                # Model Loaded
                model = charcnn.CharCNN(s=s,
                                        n_classes=config.n_classes,
                                        optimizer=config.optimizer,
                                        dims=config.embed_size,
                                        lr=config.lr,
                                        lr_decay=config.lr_decay,
                                        lr_lower_boundary=config.lr_lower_boundary)
            elif config.model == 'charrnn':
                raise NotImplementedError("[-] Not Implemented Yet")
            else:
                raise NotImplementedError("[-] Not Implemented Yet")

            # Initializing
            s.run(tf.global_variables_initializer())

            global_step = 0
            if checkpoint:
                print("[*] Reading checkpoints...")

                ckpt = tf.train.get_checkpoint_state('./model/')
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    model.saver.restore(s, ckpt.model_checkpoint_path)

                    global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                    print("[+] global step : %d" % global_step, " successfully loaded")
                else:
                    print('[-] No checkpoint file found')

            start_time = time.time()

            restored_epochs = global_step // (len(ds) // config.batch_size)
            for epoch in range(restored_epochs, config.epochs):
                for x_train, y_train in di.iterate():
                    # training
                    loss = s.run([model.opt, model.loss],
                                 feed_dict={
                                     model.x: x_train,
                                     model.y: y_train,
                                     model.do_rate: config.do_rate,
                                 })

                    if global_step % config.logging_step == 0:
                        print("[*] epoch %d global step %d" % (epoch, global_step), " loss : {:.8f}".format(loss))

                        summary = s.run([model.merged],
                                        feed_dict={
                                            model.x: x_train,
                                            model.y: y_train,
                                            model.do_rate: .0,
                                        })

                        # Summary saver
                        model.writer.add_summary(summary, global_step)

                        # Model save
                        model.saver.save(s, './model/%s.ckpt' % config.model, global_step=global_step)

                    global_step += 1

            end_time = time.time()

            print("[+] Training Done! Elapsed {:.8f}s".format(end_time - start_time))

    else:  # Test
        pass
