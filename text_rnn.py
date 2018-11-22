# -*- coding: utf-8 -*-

import tensorflow as tf


class TextRnn():
    def __init__(self, config):
        self.sequence_length = config['sequence_length']
        self.num_classes = config['num_classes']
        self.vocab_size = config['vocab_size']
        self.embedding_size = config['embedding_size']
        self.l2_reg_lambda = config['l2_reg_lambda']
        self.device = config['device']
        self.hidden_size = config['hidden_size']

        # placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # l2 loss
        l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.device(self.device), tf.name_scope('embedding'):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name='W'
            )
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        # bilstm layer
        with tf.name_scope('bi-lstm'):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            outputs, last_state = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, self.embedded_chars, dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
            self.outputs = tf.reduce_mean(outputs, axis=1)

        # add dropout
        with tf.name_scope('dropout'):
            self.rnn_drop = tf.nn.dropout(self.outputs, self.dropout_keep_prob)

        # final scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable(
                "W",
                shape=[self.hidden_size * 2, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.rnn_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda*l2_loss

        # accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
