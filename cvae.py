import tensorflow as tf
import numpy as np
import tensorflow.contrib.seq2seq as seq2seq
from numpy import mean
from tensorflow.python.layers import core as layers_core

from bleu import compute_bleu
from helpers import safe_exp, Printer

from model_helpers import Embedding, build_bidirectional_rnn, create_rnn_cell
from model_helpers import xavier

class CVAE(object):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_unit,
                 latent_dim,
                 emoji_dim,
                 batch_size,
                 kl_ceiling,
                 bow_ceiling,
                 decoder_layer=1,
                 start_i=1,
                 end_i=2,
                 beam_width=0,
                 maximum_iterations=50,
                 max_gradient_norm=5,
                 lr=1e-3,
                 dropout=0.2,
                 num_gpu=2,
                 cell_type=tf.nn.rnn_cell.GRUCell,
                 is_seq2seq=False):
        self.ori_sample = None
        self.rep_sample = None
        self.out_sample = None

        self.sess = None

        self.loss_weight = tf.placeholder_with_default(0., shape=())
        self.policy_weight = tf.placeholder_with_default(1., shape=())
        self.ac_vec = tf.placeholder(tf.float32, shape=[batch_size], name="accuracy_vector")
        self.ac5_vec = tf.placeholder(tf.float32, shape=[batch_size], name="top5_accuracy_vector")

        self.is_policy = tf.placeholder_with_default(False, shape=())
        shape = [batch_size, latent_dim]
        self.rdm = tf.placeholder_with_default(np.zeros(shape, dtype=np.float32), shape=shape)
        self.q_rdm = tf.placeholder_with_default(np.zeros(shape, dtype=np.float32), shape=shape)

        self.end_i = end_i
        self.batch_size = batch_size
        self.num_gpu = num_gpu
        self.num_unit = num_unit
        self.dropout = tf.placeholder_with_default(dropout, (), name="dropout")
        self.beam_width = beam_width
        self.cell_type = cell_type

        self.emoji = tf.placeholder(tf.int32, shape=[batch_size], name="emoji")
        self.ori = tf.placeholder(tf.int32, shape=[None, batch_size], name="original_tweet")  # [len, batch_size]
        self.ori_len = tf.placeholder(tf.int32, shape=[batch_size], name="original_tweet_length")
        self.rep = tf.placeholder(tf.int32, shape=[None, batch_size], name="response_tweet")
        self.rep_len = tf.placeholder(tf.int32, shape=[batch_size], name="response_tweet_length")
        self.rep_input = tf.placeholder(tf.int32, shape=[None, batch_size], name="response_start_tag")
        self.rep_output = tf.placeholder(tf.int32, shape=[None, batch_size], name="response_end_tag")

        self.reward = tf.placeholder(tf.float32, shape=[batch_size], name="reward")

        self.kl_weight = tf.placeholder_with_default(1., shape=(), name="kl_weight")

        self.placeholders = [
            self.emoji,
            self.ori, self.ori_len,
            self.rep, self.rep_len, self.rep_input, self.rep_output
        ]

        with tf.variable_scope("embeddings"):
            embedding = Embedding(vocab_size, embed_size)

            ori_emb = embedding(self.ori)  # [max_len, batch_size, embedding_size]
            rep_emb = embedding(self.rep)
            rep_input_emb = embedding(self.rep_input)
            emoji_emb = embedding(self.emoji)  # [batch_size, embedding_size]

        with tf.variable_scope("original_tweet_encoder"):
            ori_encoder_output, ori_encoder_state = build_bidirectional_rnn(
                num_unit, ori_emb, self.ori_len, cell_type, num_gpu, self.dropout, base_gpu=0)
            ori_encoder_state_flat = tf.concat(
                [ori_encoder_state[0], ori_encoder_state[1]], axis=1)

        emoji_vec = tf.layers.dense(emoji_emb, emoji_dim, activation=tf.nn.tanh)
        self.emoji_vec = emoji_emb
        # emoji_vec = tf.ones([batch_size, emoji_dim], tf.float32)
        condition_flat = tf.concat([ori_encoder_state_flat, emoji_vec], axis=1)

        with tf.variable_scope("response_tweet_encoder"):
            _, rep_encoder_state = build_bidirectional_rnn(
                num_unit, rep_emb, self.rep_len, cell_type, num_gpu, self.dropout, base_gpu=2)
            rep_encoder_state_flat = tf.concat(
                [rep_encoder_state[0], rep_encoder_state[1]], axis=1)

        with tf.variable_scope("representation_network"):
            rn_input = tf.concat([rep_encoder_state_flat, condition_flat], axis=1)
            # simpler representation network
            # r_hidden = rn_input
            r_hidden = tf.layers.dense(
                rn_input, latent_dim, activation=tf.nn.relu, name="r_net_hidden")  # int(1.6 * latent_dim)
            r_hidden_mu = tf.layers.dense(
                r_hidden, latent_dim, activation=tf.nn.relu)  # int(1.3 * latent_dim)
            r_hidden_var = tf.layers.dense(
                r_hidden, latent_dim, activation=tf.nn.relu)
            self.mu = tf.layers.dense(
                r_hidden_mu, latent_dim, activation=tf.nn.tanh, name="q_mean")
            self.log_var = tf.layers.dense(
                r_hidden_var, latent_dim, activation=tf.nn.tanh, name="q_log_var")

        with tf.variable_scope("prior_network"):
            # simpler prior network
            # p_hidden = condition_flat
            p_hidden = tf.layers.dense(
                condition_flat, int(0.62 * latent_dim), activation=tf.nn.relu, name="r_net_hidden")
            p_hidden_mu = tf.layers.dense(
                p_hidden, int(0.77 * latent_dim), activation=tf.nn.relu)
            p_hidden_var = tf.layers.dense(
                p_hidden, int(0.77 * latent_dim), activation=tf.nn.relu)
            self.p_mu = tf.layers.dense(
                p_hidden_mu, latent_dim, activation=tf.nn.tanh, name="p_mean")
            self.p_log_var = tf.layers.dense(
                p_hidden_var, latent_dim, activation=tf.nn.tanh, name="p_log_var")

        with tf.variable_scope("reparameterization"):
            self.normal = tf.cond(self.is_policy,
                                  lambda: self.rdm,
                                  lambda: tf.random_normal(shape=tf.shape(self.mu)))
            self.z_sample = self.mu + tf.exp(self.log_var / 2.) * self.normal

            self.q_normal = tf.cond(self.is_policy,
                                    lambda: self.q_rdm,
                                    lambda: tf.random_normal(shape=tf.shape(self.p_mu)))
            self.q_z_sample = self.p_mu + tf.exp(self.p_log_var / 2.) * self.q_normal

        if is_seq2seq:
            self.z_sample = self.z_sample - self.z_sample
            self.q_z_sample = self.q_z_sample - self.q_z_sample

        with tf.variable_scope("decoder_train") as decoder_scope:
            if decoder_layer == 2:
                train_decoder_init_state = (
                    tf.concat([self.z_sample, ori_encoder_state[0], emoji_vec], axis=1),
                    tf.concat([self.z_sample, ori_encoder_state[1], emoji_vec], axis=1)
                )
                dim = latent_dim + num_unit + emoji_dim
                cell = tf.nn.rnn_cell.MultiRNNCell(
                    [create_rnn_cell(dim, 2, cell_type, num_gpu, self.dropout),
                     create_rnn_cell(dim, 3, cell_type, num_gpu, self.dropout)])
            else:
                train_decoder_init_state = tf.concat([self.z_sample, ori_encoder_state_flat, emoji_vec], axis=1)
                dim = latent_dim + 2 * num_unit + emoji_dim
                cell = create_rnn_cell(dim, 2, cell_type, num_gpu, self.dropout)

            with tf.variable_scope("attention"):
                memory = tf.concat([ori_encoder_output[0], ori_encoder_output[1]], axis=2)
                memory = tf.transpose(memory, [1, 0, 2])

                attention_mechanism = seq2seq.LuongAttention(
                    dim, memory, memory_sequence_length=self.ori_len, scale=True)
                # attention_mechanism = seq2seq.BahdanauAttention(
                #     num_unit, memory, memory_sequence_length=self.ori_len)

            decoder_cell = seq2seq.AttentionWrapper(
                cell,
                attention_mechanism,
                attention_layer_size=dim) # TODO: add_name; what atten layer size means
            # decoder_cell = cell

            helper = seq2seq.TrainingHelper(
                rep_input_emb, self.rep_len + 1, time_major=True)
            projection_layer = layers_core.Dense(
                vocab_size, use_bias=False, name="output_projection")
            decoder = seq2seq.BasicDecoder(
                decoder_cell, helper,
                decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=train_decoder_init_state),
                output_layer=projection_layer)
            train_outputs, _, _ = seq2seq.dynamic_decode(
                decoder,
                output_time_major=True,
                swap_memory=True,
                scope=decoder_scope
            )
            self.logits = train_outputs.rnn_output

        with tf.variable_scope("decoder_infer") as decoder_scope:
            # normal_sample = tf.random_normal(shape=(batch_size, latent_dim))

            if decoder_layer == 2:
                infer_decoder_init_state = (
                    tf.concat([self.q_z_sample, ori_encoder_state[0], emoji_vec], axis=1),
                    tf.concat([self.q_z_sample, ori_encoder_state[1], emoji_vec], axis=1)
                )
            else:
                infer_decoder_init_state = tf.concat([self.q_z_sample, ori_encoder_state_flat, emoji_vec], axis=1)

            start_tokens = tf.fill([batch_size], start_i)
            end_token = end_i

            if beam_width > 0:
                infer_decoder_init_state = seq2seq.tile_batch(
                    infer_decoder_init_state, multiplier=beam_width)
                decoder = seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=embedding.coder,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=decoder_cell.zero_state(
                        batch_size * beam_width, tf.float32).clone(cell_state=infer_decoder_init_state),
                    beam_width=beam_width,
                    output_layer=projection_layer,
                    length_penalty_weight=0.0)
            else:
                helper = seq2seq.GreedyEmbeddingHelper(
                    embedding.coder, start_tokens, end_token)
                decoder = seq2seq.BasicDecoder(
                    decoder_cell,
                    helper,
                    decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=infer_decoder_init_state),
                    output_layer=projection_layer  # applied per timestep
                )

            # Dynamic decoding
            infer_outputs, _, infer_lengths = seq2seq.dynamic_decode(
                decoder,
                maximum_iterations=maximum_iterations,
                output_time_major=True,
                swap_memory=True,
                scope=decoder_scope
            )
            if beam_width > 0:
                self.result = infer_outputs.predicted_ids
            else:
                self.result = infer_outputs.sample_id
                self.result_lengths = infer_lengths

        with tf.variable_scope("loss"):
            max_time = tf.shape(self.rep_output)[0]
            with tf.variable_scope("reconstruction"):
                # TODO: use inference decoder's logits to compute recon_loss
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  # ce = [len, batch_size]
                    labels=self.rep_output, logits=self.logits)
                # rep: [len, batch_size]; logits: [len, batch_size, vocab_size]
                target_mask = tf.sequence_mask(
                    self.rep_len + 1, max_time, dtype=self.logits.dtype)
                # time_major
                target_mask_t = tf.transpose(target_mask)  # max_len batch_size
                self.recon_losses = tf.reduce_sum(cross_entropy * target_mask_t, axis=0)
                self.recon_loss = tf.reduce_sum(cross_entropy * target_mask_t) / batch_size

            with tf.variable_scope("latent"):
                # without prior network
                # self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.log_var) + self.mu ** 2 - 1. - self.log_var, 0)
                self.kl_losses = 0.5 * tf.reduce_sum(
                    tf.exp(self.log_var - self.p_log_var) +
                    (self.mu - self.p_mu) ** 2 / tf.exp(self.p_log_var) - 1. - self.log_var + self.p_log_var,
                    axis=1)
                self.kl_loss = tf.reduce_mean(self.kl_losses)

            with tf.variable_scope("bow"):
                # self.bow_loss = self.kl_weight * 0
                mlp_b = layers_core.Dense(
                    vocab_size, use_bias=False, name="MLP_b")
                # is it a mistake that we only model on latent variable?
                latent_logits = mlp_b(tf.concat(
                    [self.z_sample, ori_encoder_state_flat, emoji_vec], axis=1))  # [batch_size, vocab_size]
                latent_logits = tf.expand_dims(latent_logits, 0)  # [1, batch_size, vocab_size]
                latent_logits = tf.tile(latent_logits, [max_time, 1, 1])  # [max_time, batch_size, vocab_size]

                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  # ce = [len, batch_size]
                    labels=self.rep_output, logits=latent_logits)
                self.bow_losses = tf.reduce_sum(cross_entropy * target_mask_t, axis=0)
                self.bow_loss = tf.reduce_sum(cross_entropy * target_mask_t) / batch_size

            if is_seq2seq:
                self.kl_losses = self.kl_losses - self.kl_losses
                self.bow_losses = self.bow_losses - self.bow_losses
                self.kl_loss = self.kl_loss - self.kl_loss
                self.bow_loss = self.bow_loss - self.bow_loss

            self.losses = self.recon_losses + self.kl_losses * self.kl_weight * kl_ceiling + self.bow_losses * bow_ceiling
            self.loss = tf.reduce_mean(self.losses)

        # Calculate and clip gradients
        with tf.variable_scope("optimization"):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, max_gradient_norm)

            # Optimization
            optimizer = tf.train.AdamOptimizer(lr)
            self.update_step = optimizer.apply_gradients(
                zip(clipped_gradients, params))

        with tf.variable_scope("policy_loss"):
            prob = tf.nn.softmax(infer_outputs.rnn_output)   # [max_len, batch_size, vocab_size]
            prob = tf.clip_by_value(prob, 1e-15, 1000.)
            output_prob = tf.reduce_max(tf.log(prob), axis=2)  # [max_len, batch_size]
            seq_log_prob = tf.reduce_sum(output_prob, axis=0)  # batch_size
            # reward = tf.nn.relu(self.reward)
            self.policy_losses = - self.reward * seq_log_prob
            self.policy_losses *= (0.5 - 1) * self.ac5_vec + 1

        with tf.variable_scope("policy_optimization"):
            # zero = tf.constant(0, dtype=tf.float32)
            # where = tf.cast(tf.less(self.reward, zero), tf.float32)
            # recon = tf.reduce_sum(self.recon_losses * where) / tf.reduce_sum(where)

            final_loss = self.policy_losses * (1 - self.ac_vec) * self.policy_weight
            final_loss += self.losses * self.loss_weight
            self.policy_loss = tf.reduce_mean(final_loss)

            # final_loss = self.losses * self.loss_weight + self.policy_losses * self.policy_weight
            # final_loss *= (1 - self.ac_vec)
            # self.policy_loss = tf.reduce_sum(final_loss) / tf.reduce_sum((1 - self.ac_vec))

            gradients = tf.gradients(self.policy_loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, max_gradient_norm)
            optimizer = tf.train.AdamOptimizer(lr)
            self.policy_step = optimizer.apply_gradients(
                zip(clipped_gradients, params))

    def set_sess(self, sess):
        self.sess = sess

    def infer_and_eval(self, batches):
        a = np.random.random_integers(0, len(batches) - 1)
        b = np.random.random_integers(0, self.batch_size - 1)
        # inference
        reference_corpus = []
        generation_corpus = []

        recon_loss_l = []
        kl_loss_l = []
        bow_loss_l = []
        word_count = 0

        for index, batch in enumerate(batches):
            feed_dict = dict(zip(self.placeholders, batch))
            feed_dict[self.dropout] = 0.

            gen_digits, gen_len, recon_loss, kl_loss, bow_loss = self.sess.run(
                [self.result, self.result_lengths, self.recon_loss, self.kl_loss, self.bow_loss],
                feed_dict=feed_dict)
            recon_loss_l.append(recon_loss)
            kl_loss_l.append(kl_loss)
            bow_loss_l.append(bow_loss)

            rep_m = batch[3]
            rep_len = batch[4]
            word_count += np.sum(rep_len)
            for i, leng in enumerate(rep_len):
                ref = rep_m[0:leng, i]
                reference_corpus.append([ref])
                out = gen_digits[:gen_len[i]-1, i, 0] if self.beam_width > 0 else gen_digits[:gen_len[i]-1, i]
                generation_corpus.append(out)
                if index == a and i == b:
                    ori_m = batch[1]
                    ori_len = batch[2]
                    ori = ori_m[:ori_len[i], i]
                    self.ori_sample = ori
                    self.rep_sample = ref
                    self.out_sample = out

        total_recon_loss = np.mean(recon_loss_l)
        total_kl_loss = np.mean(kl_loss_l)
        total_bow_loss_l = np.mean(bow_loss_l)
        perplexity = safe_exp(np.sum(recon_loss_l) * self.batch_size / word_count)

        bleu_score, precisions, bp, ratio, translation_length, reference_length = compute_bleu(
            reference_corpus, generation_corpus)
        for i in range(len(precisions)):
            precisions[i] *= 100

        return (total_recon_loss, total_kl_loss, total_bow_loss_l,
                perplexity, bleu_score * 100, precisions,
                generation_corpus)

    def train_update(self, batch, weight):
        feed_dict = dict(zip(self.placeholders, batch))
        feed_dict[self.kl_weight] = weight

        _, recon_loss, kl_loss, bow_loss = self.sess.run(
            [self.update_step, self.recon_loss, self.kl_loss, self.bow_loss], feed_dict=feed_dict)
        return recon_loss, kl_loss, bow_loss

    def get_generation(self, batch, get_normal=False):
        feed_dict = dict(zip(self.placeholders, batch))
        if get_normal:
            return self.sess.run([self.result, self.result_lengths, self.normal, self.q_normal], feed_dict=feed_dict)
        else:
            return self.sess.run([self.result, self.result_lengths], feed_dict=feed_dict)

    def get_generation_for_gen(self, batch):
        feed_dict = dict(zip(self.placeholders, batch))
        return self.sess.run([self.result, self.result_lengths, self.recon_losses], feed_dict=feed_dict)

    def policy_update(self, reward, ac_vec, ac5_vec, rdm, q_rdm, batch,
                      loss_weight, policy_weight=1., kl_weight=1.):
        feed_dict = dict(zip(self.placeholders, batch))

        feed_dict[self.reward] = reward
        feed_dict[self.loss_weight] = loss_weight
        feed_dict[self.policy_weight] = policy_weight
        feed_dict[self.kl_weight] = kl_weight

        feed_dict[self.ac_vec] = ac_vec
        feed_dict[self.ac5_vec] = ac5_vec

        feed_dict[self.is_policy] = True
        feed_dict[self.rdm] = rdm
        feed_dict[self.q_rdm] = q_rdm

        _, policy_loss = self.sess.run([self.policy_step, self.policy_loss], feed_dict=feed_dict)
        return float(policy_loss)

    def policy_gen_eval(self, batches, classifier):

        gen_len_l = []
        gen_ac_l = []
        gen_ac5_l = []
        for index, batch in enumerate(batches):
            emoji, ori, ori_len, rep, rep_len, _, _ = batch
            gen, gen_len = self.get_generation(batch)
            with classifier.sess.graph.as_default():
                gen_prob, gen_ac, gen_ac5 = classifier.get_prob(emoji, gen, gen_len)
            gen_len_l.append(mean(gen_len))
            gen_ac_l.append(mean(gen_ac.astype(np.float32)))
            gen_ac5_l.append(mean(gen_ac5.astype(np.float32)))
        return mean(gen_len_l), mean(gen_ac_l), mean(gen_ac5_l)
