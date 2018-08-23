import numpy as np
import tensorflow as tf


class CharCNN:

    def __init__(self, s, n_classes=10, batch_size=128, epochs=100,
                 vocab_size=251, dims=300, seed=1337, use_d2v=True,
                 filter_sizes=(1, 2, 3, 4), n_filters=256, fc_unit=1024,
                 lr=5e-4, lr_lower_boundary=1e-5, lr_decay=.95, l2_reg=1e-3, th=1e-6):
        self.s = s
        self.n_dims = dims
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.use_d2v = use_d2v
        self.seed = seed

        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.fc_unit = fc_unit
        self.l2_reg = l2_reg
        self.th = th

        # set random seed
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.he_uni = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
        self.reg = tf.contrib.layers.l2_regularizer(self.l2_reg)

        if not use_d2v:  # use random initialization # use w2v initialization
            self.embeddings = tf.get_variable('lookup-w', shape=[self.vocab_size, self.n_dims],
                                              initializer=self.he_uni)

        self.x = tf.placeholder(tf.float32, shape=[None, self.n_dims], name='x-sentence')
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='y-label')
        self.do_rate = tf.placeholder(tf.float32, name='do-rate')

        # build CharCNN Model
        self.rate = self.build_model()

        # loss
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(
            labels=self.y,
            predictions=self.rate
        ))

        # Optimizer
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(lr,
                                                   self.global_step,
                                                   self.epochs * self.batch_size,
                                                   lr_decay,
                                                   staircase=True)
        self.lr = tf.clip_by_value(learning_rate,
                                   clip_value_min=lr_lower_boundary,
                                   clip_value_max=1e-3,
                                   name='lr-clipped')

        self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr).minimize(self.loss)

        # Mode Saver/Summary
        tf.summary.scalar('loss', self.loss)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)

    def build_model(self):
        if not self.use_d2v:
            with tf.name_scope('embeddings'):
                spatial_do = tf.contrib.keras.layers.SpatialDropout1D(self.do_rate)

                embeds = tf.nn.embedding_lookup(self.embeddings, self.x)
                embeds = spatial_do(embeds)
        else:
            embeds = self.x

        pooled_outs = []
        for i, fs in enumerate(self.filter_sizes):
            with tf.variable_scope("conv_layer-%d-%d" % (fs, i)):
                """
                Conv1D-ThresholdReLU-drop_out-k_max_pool
                """

                x = tf.layers.conv1d(
                    embeds,
                    filters=self.n_filters,
                    kernel_size=fs,
                    kernel_initializer=self.he_uni,
                    kernel_regularizer=self.reg,
                    padding='VALID',
                    name='conv1d'
                )
                x = tf.where(tf.less(x, self.th), tf.zeros_like(x), x)
                x = tf.layers.dropout(x, self.do_rate)

                x = tf.nn.top_k(tf.transpose(x, [0, 2, 1]), k=3, sorted=False)[0]
                x = tf.transpose(x, [0, 2, 1])

                pooled_outs.append(x)

        x = tf.concat(pooled_outs, 1)
        x = tf.layers.flatten(x)
        x = tf.layers.dropout(x, self.do_rate)

        with tf.variable_scope("outputs"):
            x = tf.layers.dense(
                x,
                units=self.fc_unit,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='fc1'
            )
            x = tf.where(tf.less(x, self.th), tf.zeros_like(x), x)

            x = tf.layers.dense(
                x,
                units=self.n_classes,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='fc2'
            )
            rate = tf.nn.sigmoid(x)  # To-Do : replace with another scale function to avoid saturation
            rate = rate * 9. + 1.    # 1 ~ 10
            return rate
