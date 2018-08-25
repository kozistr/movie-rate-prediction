import time
import argparse
import numpy as np
import tensorflow as tf

from model import charcnn
from dataloader import Doc2VecEmbeddings, DataLoader, DataIterator


parser = argparse.ArgumentParser(description='train/test movie review classification model')
parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument('--model_name', type=str, help='classification model', default='charcnn')
parser.add_argument('--dataset', type=str, help='DataSet path', default='tagged_data.csv')
parser.add_argument('--n_threads', type=int, help='the number of threads', default=8)
parser.add_argument('--model', type=str, help='trained w2v/d2v model file', default='ko_d2v.model')
parser.add_argument('--n_dims', type=int, help='embeddings'' dimensions', default=300)
parser.add_argument('--n_classes', type=int, help='the number of classes', default=10)
parser.add_argument('--seed', type=int, help='random seed', default=1337)
parser.add_argument('--save_to_file', type=bool, help='save DataSet into .csv file', default=True)
parser.add_argument('--save_file', type=str, help='DataSet file name', default='tagged_data.csv')
parser.add_argument('--verbose', type=bool, help='print progress', default=True)
args = parser.parse_args()

# parsed args
mode = args.mode
seed = args.seed
n_dims = args.n_dims
dataset = args.dataset
vec_model = args.model
verbose = args.verbose
n_classes = args.n_classes
n_threads = args.n_threads
model_name = args.model_name

save_to_file = args.save_to_file
save_file = args.save_file

np.random.seed(seed)
tf.set_random_seed(seed)


if __name__ == '__main__':
    # DataSet Loader
    ds = DataLoader(dataset,
                    n_classes=n_classes,
                    is_tagged_file=True,
                    save_to_file=save_to_file,
                    use_in_time_save=False,
                    # save_file=save_file,
                    n_threads=n_threads)

    if verbose:
        print("[+] DataSet loaded! Total %d samples" % len(ds.labels))

    # Doc2Vec Loader
    vec = Doc2VecEmbeddings(vec_model, n_dims)
    if verbose:
        print("[+] Doc2Vec loaded! Total %d pre-trained sentences, %d dims" % (len(vec), n_dims))

    if mode == 'train':
        # GPU configure
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as s:
            if model_name == 'charcnn':
                # Model Loaded
                model = charcnn.CharCNN(s=s,
                                        n_classes=n_classes,
                                        dims=n_dims)
            else:
                raise NotImplementedError("[-] Not Implemented Yet")

            # Initializing
            s.run(tf.global_variables_initializer())

            # DataSet Iterator
            di = DataIterator(x=ds.sentences, y=ds.labels,
                              batch_size=model.batch_size)

            # To-Do
            # pre-trained model loader

            start_time = time.time()

            epochs = 10
            global_step = 0
            logging_step = 1000
            for epoch in range(epochs):
                for x_train, y_train in di.iterate():
                    # training
                    loss = s.run([model.opt, model.loss],
                                 feed_dict={
                                     model.x: x_train,
                                     model.y: y_train,
                                     model.do_rate: .8,
                                 })

                    if global_step % logging_step == 0:
                        print("[*] epoch %d step %d" % (epoch, global_step), " loss : {:.8f}" % loss)

                        summary = s.run([model.merged],
                                        feed_dict={
                                            model.x: x_train,
                                            model.y: y_train,
                                            model.do_rate: .0,
                                        })

                        # Summary saver
                        model.writer.add_summary(summary, global_step)

                        # Model save
                        model.saver.save(s, './model/%s.ckpt' % model_name, global_step=global_step)

                    global_step += 1

            end_time = time.time()

            print("[+] Training Done! Elapsed {:.8f}s".format(end_time - start_time))

    elif mode == 'test':
        pass
    else:
        print('[-] mode should be train or test')
