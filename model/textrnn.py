import numpy as np
import tensorflow as tf


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


class TextRNN:

    def __init__(self, s, n_classes=10, batch_size=128, epochs=100,
                 vocab_size=122351 + 1, sequence_length=400, n_dims=300, seed=1337, optimizer='adam',
                 n_gru_layers=2, n_gru_cells=256, n_attention_size=128, fc_unit=1024,
                 lr=5e-4, lr_lower_boundary=1e-5, lr_decay=.9, l2_reg=5e-4, th=1e-6,
                 summary=None, mode='static', w2v_embeds=None):
        self.s = s
        self.n_dims = n_dims
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        self.batch_size = batch_size
        self.epochs = epochs

        self.seed = seed

        self.n_gru_layers = n_gru_layers
        self.n_gru_cells = n_gru_cells
        self.n_attention_size = n_attention_size
        self.fc_unit = fc_unit
        self.l2_reg = l2_reg
        self.th = th

        self.optimizer = optimizer

        self.summary = summary
        self.mode = mode
        self.w2v_embeds = w2v_embeds

        # set random seed
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.he_uni = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
        self.reg = tf.contrib.layers.l2_regularizer(self.l2_reg)

        if self.mode == 'static':
            # uncompleted feature
            self.embeddings = tf.get_variable('embeddings', shape=[self.vocab_size, self.n_dims],
                                              initializer=self.he_uni, trainable=False)
        elif self.mode == 'non-static' or self.mode == 'rand':
            self.embeddings = tf.get_variable('embeddings', shape=[self.vocab_size, self.n_dims],
                                              initializer=self.he_uni, trainable=True)
        else:
            raise NotImplementedError("[-] static or non-static or rand only! (%s)" % self.mode)

        if not self.mode == 'rand':
            assert self.w2v_embeds is not None
            self.embeddings = self.embeddings.assign(self.w2v_embeds)

            print("[+] Word2Vec pre-trained model loaded!")

        self.x = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name='x-sentence')
        self.y = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='y-label')  # one-hot or int
        self.do_rate = tf.placeholder(tf.float32, name='do-rate')

        # build CharCNN Model
        self.feat, self.rate = self.build_model()

        # loss
        if self.n_classes == 1:
            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(
                labels=self.y,
                predictions=self.rate
            ))  # MSE loss

            self.prediction = self.rate
            self.accuracy = tf.reduce_mean(tf.cast((tf.abs(self.y - self.prediction) < .5), dtype=tf.float32))
        else:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.feat,
                labels=self.y
            ))  # softmax cross-entropy

            self.prediction = tf.argmax(self.rate, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), self.prediction), dtype=tf.float32))

        # Optimizer
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(lr,
                                                   self.global_step,
                                                   100000,  # hard-coded
                                                   lr_decay,
                                                   staircase=True)
        self.lr = tf.clip_by_value(learning_rate,
                                   clip_value_min=lr_lower_boundary,
                                   clip_value_max=1e-3,
                                   name='lr-clipped')

        if self.optimizer == 'adam':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr).minimize(self.loss)
        elif self.optimizer == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        elif self.optimizer == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr).minimize(self.loss)
        else:
            raise NotImplementedError("[-] only Adam, SGD are supported!")

        # Mode Saver/Summary
        tf.summary.scalar('loss/loss', self.loss)
        tf.summary.scalar('misc/lr', self.lr)
        tf.summary.scalar('misc/acc', self.accuracy)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model savers
        self.saver = tf.train.Saver(max_to_keep=1)
        self.best_saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.summary, self.s.graph)

    def build_model(self):
        with tf.device('/cpu:0'), tf.name_scope('embeddings'):
            spatial_drop_out = tf.keras.layers.SpatialDropout1D(self.do_rate)

            embeds = tf.nn.embedding_lookup(self.embeddings, self.x)
            embeds = spatial_drop_out(embeds)

        x = embeds
        outs = []

        with tf.name_scope("cudnnGRU"):
            gru = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=self.n_gru_layers, num_units=self.n_gru_cells,
                                                direction='bidirectional',
                                                seed=self.seed, kernel_initializer=self.he_uni, name='bigru1')
            x, _ = gru(x)  # (?, 140, 512)

        # 1. lambda : get last hidden state
        outs.append(tf.reshape(x[:, -1, :], (-1, x.get_shape()[-1])))  # (?, 512)

        # 2. GlobalMaxPooling1d
        outs.append(tf.reduce_max(x, axis=1))  # (?, 512)

        # 3. GlobalAvgPooling1d
        outs.append(tf.reduce_mean(x, axis=1))  # (?, 512)

        # 4. AttentionWeightedAverage
        outs.append(attention(x, self.n_attention_size))  # (?, 512)

        x = tf.concat(outs, axis=-1)
        x = tf.layers.flatten(x)  # (?, 2048)
        x = tf.layers.dropout(x, self.do_rate)

        with tf.variable_scope("outputs"):
            x = tf.layers.dense(
                x,
                units=self.fc_unit,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='fc1'
            )
            x = tf.nn.relu(x)

            x = tf.layers.dense(
                x,
                units=self.n_classes,
                kernel_initializer=self.he_uni,
                kernel_regularizer=self.reg,
                name='fc2'
            )

            # Rate
            if self.n_classes == 1:
                rate = tf.nn.sigmoid(x)
                rate = rate * 9. + 1.  # To-Do : replace with another scale function to avoid saturation
            else:
                rate = tf.nn.softmax(x)
            return x, rate
