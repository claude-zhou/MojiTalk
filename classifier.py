import tensorflow as tf
import numpy as np

from emoji_reader import emoji_64
from model_helpers import Embedding, xavier, build_bidirectional_rnn

class EmojiClassifier(object):
    def __init__(self,
                 batch_size,
                 vocab_size,
                 emoji_num,
                 embed_size,
                 num_unit,
                 num_gpu,
                 lr=0.001,
                 dropout=0.2,
                 cell_type=tf.nn.rnn_cell.GRUCell
                 ):
        self.batch_size = batch_size
        self.sess = None
        self.emoji_index = None

        self.dropout = tf.placeholder_with_default(dropout, (), name="dropout")
        self.num_gpu = num_gpu
        self.cell_type = cell_type

        self.text = tf.placeholder(tf.int32, shape=[None, batch_size], name="text")
        self.len = tf.placeholder(tf.int32, shape=[batch_size], name="text_length")
        self.emoji = tf.placeholder(tf.int32, shape=[batch_size], name="emoji_label")

        with tf.variable_scope("embeddings"):
            embedding = Embedding(vocab_size, embed_size)
            text_emb = embedding(self.text)

        with tf.variable_scope("bi_rnn_1"):  # difference between var scope and name scope?
            # tuple#2: [max_time, batch_size, num_unit]
            outputs_1, _ = build_bidirectional_rnn(
                num_unit, text_emb, self.len, cell_type, num_gpu, drop=self.dropout, base_gpu=0)

        with tf.variable_scope("bi_rnn_2"):
            rnn2_input = tf.concat([outputs_1[0], outputs_1[1]], axis=2)
            outputs_2, _ = build_bidirectional_rnn(
                num_unit, rnn2_input, self.len, cell_type, num_gpu, drop=self.dropout, base_gpu=2)

        with tf.variable_scope("attention"):
            word_states = tf.concat(  # [outputs_1[0], outputs_1[1], text_emb], axis=2)
                [outputs_1[0], outputs_1[1], outputs_2[0], outputs_2[1], text_emb], axis=2)  # [max_t, b_sz, h_dim]

            weights = tf.layers.dense(word_states, 1)
            weights = tf.exp(weights)   # [max_len, batch_size, 1]

            # mask superfluous dimensions
            max_time = tf.shape(self.text)[0]
            weight_mask = tf.sequence_mask(self.len, max_time, dtype=tf.float32)
            weight_mask = tf.expand_dims(
                tf.transpose(weight_mask), axis=-1)  # transpose for time_major & expand to be broadcast-able
            weights = weights * weight_mask

            # weight regularization
            sums = tf.expand_dims(tf.reduce_sum(weights, axis=0), 0)  # [1, batch_size, 1]
            sums = tf.clip_by_value(sums, 1e-10, 1000.)
            weights = weights / sums

            weights = tf.transpose(weights, [1, 0, 2])  # [batch_size, max_len, 1]
            word_states = tf.transpose(word_states, [1, 2, 0])  # [batch_size, h_dim, max_len]
            text_vec = tf.squeeze(tf.matmul(word_states, weights), axis=2)  # [batch_size, h_dim]

        with tf.variable_scope("loss"):
            self.logits = tf.layers.dense(text_vec, emoji_num)
            self.prob = tf.nn.softmax(self.logits)
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.emoji, logits=self.logits))

        with tf.variable_scope("accuracy"):
            self.top_5_accuracy_vec = tf.nn.in_top_k(self.logits, self.emoji, k=5)
            self.top_5_accuracy = tf.reduce_mean(tf.cast(self.top_5_accuracy_vec, tf.float32))

            self.accuracy_vec = tf.nn.in_top_k(self.logits, self.emoji, k=1)
            self.accuracy = tf.reduce_mean(tf.cast(self.accuracy_vec, tf.float32))

        with tf.variable_scope("optimization"):
            optimizer = tf.train.AdamOptimizer(lr)
            self.update_step = optimizer.minimize(self.loss)

    def set_sess(self, sess):
        self.sess = sess

    def set_emoji_index(self, emoji_index):
        self.emoji_index = emoji_index

    def train_update(self, batch):

        text = batch[3]
        emoji_label = map_emoji(batch[0], self.emoji_index)
        length = batch[4]

        _, loss, accuracy, accuracy5 = self.sess.run(
            [self.update_step, self.loss, self.accuracy, self.top_5_accuracy],
            feed_dict={self.text: text, self.emoji: emoji_label, self.len: length})
        return loss, accuracy, accuracy5

    def eval(self, batches):

        loss_l = []
        accuracy_l = []
        accuracy5_l = []

        for batch in batches:
            text = batch[3]
            emoji_label = map_emoji(batch[0], self.emoji_index)
            length = batch[4]

            loss, accuracy, accuracy5 = self.sess.run(
                [self.loss, self.accuracy, self.top_5_accuracy],
                feed_dict={self.text: text, self.emoji: emoji_label, self.len: length, self.dropout: 0.})

            loss_l.append(loss)
            accuracy_l.append(accuracy)
            accuracy5_l.append(accuracy5)
        return float(np.mean(loss_l)), float(np.mean(accuracy_l)), float(np.mean(accuracy5_l))

    def eval_2(self, batches):
        from collections import Counter
        total = Counter()
        hit = Counter()
        hit5 = Counter()

        for batch in batches:
            text = batch[3]
            emoji_label = map_emoji(batch[0], self.emoji_index)
            length = batch[4]
            total.update(list(emoji_label))

            accuracy, accuracy5 = self.sess.run(
                [self.accuracy_vec, self.top_5_accuracy_vec],
                feed_dict={self.text: text, self.emoji: emoji_label, self.len: length, self.dropout: 0.})

            for i in range(self.batch_size):
                if accuracy[i]:
                    hit.update([emoji_label[i]])
                if accuracy5[i]:
                    hit5.update([emoji_label[i]])
        return total, hit, hit5

    def get_prob(self, emoji, gen, gen_len):

        emoji = map_emoji(emoji, self.emoji_index) # b2s
        row_index = np.asarray(range(emoji.shape[0]))
        # index = np.stack([row_index, emoji], axis=-1)
        prob, ac_vec, ac5_vec = self.sess.run(
            [self.prob, self.accuracy_vec, self.top_5_accuracy_vec],
            feed_dict={self.text: gen, self.emoji: emoji, self.len: gen_len})
        prob = prob[row_index, emoji]
        return prob, ac_vec, ac5_vec

    def get_prob_for_gen(self, emoji_no_map, gen, gen_len):

        emoji = emoji_no_map
        row_index = np.asarray(range(emoji.shape[0]))
        # index = np.stack([row_index, emoji], axis=-1)
        prob, ac_vec, ac5_vec = self.sess.run(
            [self.prob, self.accuracy_vec, self.top_5_accuracy_vec],
            feed_dict={self.text: gen, self.emoji: emoji, self.len: gen_len})
        prob = prob[row_index, emoji]
        return prob, ac_vec, ac5_vec

    def get_logits(self, emoji, gen, gen_len):

        emoji = map_emoji(emoji, self.emoji_index)
        loss, logits, ac, ac5 = self.sess.run(
            [self.loss, self.logits, self.accuracy, self.top_5_accuracy],
            feed_dict={self.text: gen, self.emoji: emoji, self.len: gen_len})
        return loss, logits, ac, ac5

    def get_all_prob(self, batches):
        p = None
        for batch in batches:
            text = batch[3]
            emoji_label = map_emoji(batch[0], self.emoji_index)
            length = batch[4]

            prob = self.sess.run(
                self.prob,
                feed_dict={self.text: text, self.emoji: emoji_label, self.len: length, self.dropout: 0.})
            if p is None:
                p = prob
            else:
                p = np.concatenate((p, prob), axis=0)
        return p

    def get_all_prob_and_eval(self, batches):
        p = None
        loss_l = []
        accuracy_l = []
        accuracy5_l = []
        for batch in batches:
            text = batch[3]
            emoji_label = map_emoji(batch[0], self.emoji_index)
            length = batch[4]

            loss, prob, accuracy, accuracy5 = self.sess.run(
                [self.loss, self.prob, self.accuracy, self.top_5_accuracy],
                feed_dict={self.text: text, self.emoji: emoji_label, self.len: length, self.dropout: 0.})
            p = prob if p is None else np.concatenate((p, prob), axis=0)

            loss_l.append(loss)
            accuracy_l.append(accuracy)
            accuracy5_l.append(accuracy5)
        return p, float(np.mean(loss_l)), float(np.mean(accuracy_l)), float(np.mean(accuracy5_l))


emoji_num = 64
from params.full import *

def map_emoji(word_indices, emoji_index_dict):
    return np.array([emoji_index_dict[index] for index in word_indices])


if __name__ == '__main__':

    from time import gmtime, strftime
    from os import makedirs, chdir
    from os.path import join, dirname
    import json

    from helpers import build_vocab, build_data, batch_generator, build_emoji_index
    from helpers import print_out

    get_down_params = False

    num_epoch = 6
    test_step = 50

    chdir("mojitalk_data")
    output_dir = join("classifier", strftime("%m-%d_%H-%M-%S", gmtime()))

    vocab_f = "vocab.ori"
    train_ori_f = "train.ori"
    train_rep_f = "train.rep"
    test_ori_f = "test.ori"  # "dev.ori"
    test_rep_f = "test.rep"  # "dev.rep"

    makedirs(dirname(join(output_dir, "breakpoints/")), exist_ok=True)
    log_f = open(join(output_dir, "log.txt"), "w")
    emoji_index, _, emoji_sorted = build_emoji_index(vocab_f, emoji_64)

    # build vocab
    word2index, index2word = build_vocab(vocab_f)
    start_i, end_i = word2index['<s>'], word2index['</s>']
    vocab_size = len(word2index)

    classifier = EmojiClassifier(batch_size, vocab_size, emoji_num, embed_size, num_unit, num_gpu)

    # build data
    train_data = build_data(train_ori_f, train_rep_f, word2index)
    test_data = build_data(test_ori_f, test_rep_f, word2index)
    test_batches = batch_generator(
        test_data, start_i, end_i, batch_size, permutate=False)

    print_out("*** CLASSIFIER DATA READY ***")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        classifier.set_sess(sess)
        classifier.set_emoji_index(emoji_index)

        global_step = best_step = 1
        start_epoch = best_epoch = 1
        best_loss = 1000.
        sess.run(tf.global_variables_initializer())

        for epoch in range(start_epoch, num_epoch + 1):
            train_batches = batch_generator(
                train_data, start_i, end_i, batch_size)

            loss_l = []
            accuracy_l = []
            accuracy5_l = []
            for batch in train_batches:
                loss, accuracy, accuracy5 = classifier.train_update(batch)
                loss_l.append(loss)
                accuracy_l.append(accuracy)
                accuracy5_l.append(accuracy5)

                if global_step % test_step == 0:
                    time_now = strftime("%m-%d %H:%M:%S", gmtime())
                    print_out('epoch:\t%d\tstep:\t%d\tbatch-loss/accuracy/accuracy5:\t%.3f\t%.1f\t%.1f\t\t%s' %
                              (epoch, global_step,
                               np.mean(loss_l), np.mean(accuracy_l) * 100, np.mean(accuracy5_l) * 100, time_now),
                              f=log_f)
                if global_step % (test_step * 10) == 0:
                    loss, accuracy, accuracy5 = classifier.eval(test_batches)
                    print_out('EPOCH-\t%d\tSTEP-\t%d\tTEST-loss/accuracy/accuracy5-\t%.3f\t%.1f\t%.1f' %
                              (epoch, global_step,
                               loss, accuracy * 100, accuracy5 * 100),
                              f=log_f)

                    if best_loss >= loss:
                        best_loss = loss

                        best_epoch = epoch
                        best_step = global_step

                        # save breakpoint
                        path = join(output_dir, "breakpoints/best_test_loss.ckpt")
                        save_path = saver.save(sess, path)

                        # save best epoch/step
                        best_dict = {
                            "loss": best_loss, "epoch": best_epoch, "step": best_step, "accuracy": accuracy,
                            "top_5_accuracy": accuracy5}
                        with open(path, "w") as f:
                            f.write(json.dumps(best_dict, indent=2))
                global_step += 1

            if get_down_params:
                prob = classifier.get_all_prob(test_batches)    # batch_num * 64
                cov = np.corrcoef(prob, rowvar=False)
                with open("test%d.npy" % epoch, "wb") as f:
                    np.save(f, cov)
                prob = classifier.get_all_prob(train_batches)   # batch_num * 64
                cov = np.corrcoef(prob, rowvar=False)
                with open("train%d.npy"% epoch, "wb") as f:
                    np.save(f, cov)

    log_f.close()
