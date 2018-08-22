import numpy as np
import tensorflow as tf


class CharCNN:

    def __init__(self, n_classes=10, batch_size=128,
                 vocab_size=251, dims=300, seed=1337, use_w2v=False, w2v_model=None,
                 filter_size=(1, 2, 3, 4), n_filters=256, fc_unit=1024,
                 l2_reg=1e-3)
        self.n_dims = dims
        self.n_classes = n_classes
        self.w2v_model = w2v_model
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.seed = seed
        self.use_w2v = use_w2v

        if use_w2v:
            assert w2v_model

        self.filter_size = filter_size
        self.n_filters = n_filters
        self.fc_unit = fc_unit
        self.l2_reg = l2_reg

        # set random seed
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.he_uni = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
        self.reg = tf.contrib.layers.l2_regularizer(self.l2_reg)

        if use_w2v:  # use w2v initialization
            self.embeddings = tf.Variable(initial_value=tf.constant(self.w2v_model),
                                          trainable=False)
        else:
            self.embeddings = tf.get_variable('lookup-w', shape=[self.vocab_size, self.n_dims],
                                              initializer=self.he_uni)

        self.x = tf.placeholder(tf.float32, shape=[None, self.n_dims], name='x-sentence')
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='y-label')
        self.do_rate = tf.placeholder(tf.float32, name='do-rate')

        # build CharCNN Model
        self.build_model()

    def build_model(self):
        with tf.name_scope('embeddings'):
            spatial_do = tf.contrib.keras.layers.SpatialDropout1D(self.do_rate)

            embeds = tf.nn.embedding_lookup(self.embeddings, self.x)
            embeds = spatial_do(embeds)

        pooled_outs = []
        for i, fs in enumerate(self.filter_size):
            pass


if __name__ == '__main__':
    pass
