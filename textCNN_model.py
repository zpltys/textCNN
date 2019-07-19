# -*- coding: utf-8 -*-
#TextCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer.
import tensorflow as tf
import numpy as np

class TextCNN:
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,vocab_size, embed_size,
                 initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0, decay_rate_big=0.50, use_mulitple_layer_cnn=False):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate", dtype=tf.float32)
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes)
        self.clip_gradients = clip_gradients
        self.is_training_flag = tf.placeholder(tf.bool, name="is_training_flag")

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.iter = tf.placeholder(tf.int32) #training iteration
        self.tst = tf.placeholder(tf.bool)
        self.use_mulitple_layer_cnn = use_mulitple_layer_cnn

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign_add(self.epoch_step, tf.constant(1))
        self.b1 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.b2 = tf.Variable(tf.ones([self.num_filters]) / 10)
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference()
        self.possibility = tf.nn.sigmoid(self.logits)

        self.loss_val = self.loss()
        self.train_op = self.train()


    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection",shape=[self.num_filters_total, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size] #ADD 2017.06.09

    def inference(self):
        """main computation graph here: 1.embedding-->2.CONV-BN-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)#[None,sentence_length,embed_size]
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)

        if self.use_mulitple_layer_cnn:
            print("use multi layer CNN")
            h = self.cnn_multiple_layers()
        else:
            print("use single layer CNN")
            h = self.cnn_single_layer()

        with tf.name_scope("output"):
            logits = tf.matmul(h, self.W_projection) + self.b_projection
        return logits

    def cnn_single_layer(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("convolution-pooling-%s" % filter_size):
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters], initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn_bn_')

                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID',name="pool")
                pooled_outputs.append(pooled)

        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1,self.num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]
        h = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)
        return h

    def cnn_multiple_layers(self):
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('cnn_multiple_layers' + "convolution-pooling-%s" % filter_size):
                print(i, "sentence_embeddings_expanded:", self.sentence_embeddings_expanded)
                # 1) CNN->BN->relu
                filter1 = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],initializer=self.initializer)
                conv1 = tf.nn.conv2d(self.sentence_embeddings_expanded, filter1, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                conv1 = tf.contrib.layers.batch_norm(conv1, is_training=self.is_training_flag, scope='cnn1')
                print(i, "conv1:", conv1)
                b1 = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), "relu")  # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 2) CNN->BN->relu
                h2 = tf.reshape(h1, [-1, self.sequence_length - filter_size + 1, self.num_filters, 1])  # shape:[batch_size,sequence_length,num_filters,1]
                # Layer2:CONV-RELU
                filter2 = tf.get_variable("filter2-%s" % filter_size, [filter_size, self.num_filters, 1, self.num_filters], initializer=self.initializer)
                conv2 = tf.nn.conv2d(h2, filter2, strides=[1, 1, 1, 1], padding="VALID", name="conv2")  # shape:[batch_size,sequence_length,1,num_filters]
                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_training_flag, scope='cnn2')
                print(i, "conv2:", conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])
                h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), "relu2")

                # 3. Max-pooling
                pooling_max = tf.nn.max_pool(h2, ksize=[1, self.sequence_length - 2 * filter_size + 2, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                print(i, "pooling:", pooling_max)
                pooling_max = tf.reshape(pooling_max, [-1, self.num_filters])
                print(i, "pooling:", pooling_max)
                pooled_outputs.append(pooling_max)  # h:[batch_size,sequence_length,1,num_filters]
        # concat
        h = tf.concat(pooled_outputs, axis=1)  # [batch_size,num_filters*len(self.filter_sizes)]
        print("h.concat:", h)

        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)

        h = tf.layers.dense(h, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)
        return h

    def loss(self, l2_lambda=0.001): #0.0001#this loss function is for multi-label classification
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:", losses)
            losses = tf.reduce_sum(losses,axis=1)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

 #   def train(self):
 #       self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
#                                                   self.decay_rate, staircase=False)
   #     train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val, global_step=self.global_step)
   #     return train_op

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate_ = learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op
