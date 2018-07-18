import tensorflow as tf
import os
from os import makedirs
from os.path import join, dirname
from time import strftime, gmtime
import numpy as np
from numpy import mean
from math import tanh
import json

from helpers import *
from cvae import CVAE
from classifier import EmojiClassifier
from emoji_reader import emoji_64

"""hyper-params"""
from params.full import *
num_epoch = 6
test_step = 20
emoji_num = 64
num_gpu = 1

"""directories"""
os.chdir("mojitalk_data")
output_dir = join("reinforced_cvae", strftime("%m-%d_%H-%M-%S", gmtime()))

vocab_f = "vocab.ori"
train_ori_f = "train.ori"
train_rep_f = "train.rep"
test_ori_f = "test.ori"     # dev.ori
test_rep_f = "test.rep"     # dev.rep

train_out_f = join(output_dir, "train.out")
test_out_f = join(output_dir, "test.out")   # dev.out

makedirs(dirname(join(output_dir, "breakpoints/")), exist_ok=True)
log_f = open(join(output_dir, "log.txt"), "w", encoding="utf-8")

"""build vocabulary"""
word2index, index2word = build_vocab(vocab_f)
start_i, end_i = word2index['<s>'], word2index['</s>']
vocab_size = len(word2index)
emoji_b2s, emoji_s2b, emoji_sorted = build_emoji_index(vocab_f, emoji_64)

p = Printer(log_f, index2word)

"""build graphs and init params"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
graph0 = tf.Graph()
with graph0.as_default():
    kl_ceiling = 0.48
    seq2seq = CVAE(vocab_size, embed_size, num_unit, latent_dim, emoji_dim, batch_size,
                   kl_ceiling, 1, decoder_layer,
                   start_i, end_i, beam_width, maximum_iterations, max_gradient_norm, lr, dropout, num_gpu,
                   cell_type,
                   is_seq2seq=False)
    sess0 = tf.Session(graph=graph0, config=config)
    sess0.run(tf.global_variables_initializer())
    seq2seq.set_sess(sess0)
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver0 = tf.train.Saver(var_list=train_vars, max_to_keep=25)
    saver0.restore(sess0, "cvae/07-17_15-51-04/breakpoints/at_step_36500.ckpt")
graph0.finalize()

graph1 = tf.Graph()
with graph1.as_default():
    classifier = EmojiClassifier(batch_size, vocab_size, emoji_num, embed_size, num_unit, num_gpu,
                                 lr=0.001, dropout=0.2, cell_type=tf.nn.rnn_cell.GRUCell)
    saver1 = tf.train.Saver()
    sess1 = tf.Session(graph=graph1, config=config)
    classifier.set_sess(sess1)
    classifier.set_emoji_index(emoji_b2s)
    saver1.restore(sess1, "classifier/07-16_14-33-58/breakpoints/best_test_loss.ckpt")
graph1.finalize()

"""build data"""
test_data = build_data(test_ori_f, test_rep_f, word2index)
test_batches = batch_generator(test_data, start_i, end_i, batch_size, permutate=False)
train_data = build_data(train_ori_f, train_rep_f, word2index)


global_step = best_step = 1
start_epoch = best_epoch = 1

total_step = (8 * len(train_data[0]) / batch_size)

(test_recon_loss, test_kl_loss, test_bow_loss,
 perplexity, test_bleu_score, precisions, _) = seq2seq.infer_and_eval(test_batches)
p.put_example(seq2seq)
p.put_bleu(
    test_recon_loss, test_kl_loss, test_bow_loss,
    perplexity, test_bleu_score, precisions, "TEST")


lengths, ac, ac5 = seq2seq.policy_gen_eval(test_batches, classifier)
p.put_list([lengths, ac, ac5])

s = 1.
e = 1.
step = 500

for epoch in range(start_epoch, num_epoch + 1):
    train_batches = batch_generator(train_data, start_i, end_i, batch_size, permutate=True)

    l = [[] for i in range(5)]
    for batch in train_batches:
        emoji, ori, ori_len, rep, rep_len, _, _ = batch
        gen, gen_len, normal, q_normal = seq2seq.get_generation(batch, get_normal=True)
        with graph1.as_default():
            gen_prob, gen_ac, gen_ac5 = classifier.get_prob(emoji, gen, gen_len)
            tar_prob, tar_ac, tar_ac5 = classifier.get_prob(emoji, rep, rep_len)

            gen_ac = gen_ac.astype(np.float32)
            gen_ac5 = gen_ac5.astype(np.float32)
            tar_ac = tar_ac.astype(np.float32)
            tar_ac5 = tar_ac5.astype(np.float32)

            reward = gen_prob - tar_prob
        kl_w = get_kl_weight(global_step + 9500, total_step, 0.75)
        pl_w = 1.05
        policy_loss = seq2seq.policy_update(
            reward, gen_ac, gen_ac5, normal, q_normal, batch,
            min((e - s) / step * global_step + s, e),
            pl_w,
            kl_w)

        reward *= (0.5 - 1) * gen_ac5 + 1
        reward *= 1 - gen_ac

        l[0].append(mean(gen_len))
        l[1].append(mean(tar_ac) - mean(gen_ac))
        l[2].append(mean(tar_ac5) - mean(gen_ac5))
        l[3].append(np.sum(reward))
        l[4].append(policy_loss)

        if global_step % test_step == 0:
            l = [mean(ll) for ll in l]
            p.put_step(epoch, global_step)
            p.put_list(l)
            l = [[] for i in range(5)]

        if global_step % (test_step * 20) == 10 * test_step:
            (test_recon_loss, test_kl_loss, test_bow_loss,
             perplexity, test_bleu_score, precisions, _) = seq2seq.infer_and_eval(test_batches)
            p.put_example(seq2seq)
            p.put_bleu(
                test_recon_loss, test_kl_loss, test_bow_loss,
                perplexity, test_bleu_score, precisions, "TEST")

        if global_step % (test_step * 10) == 0:
            p.put_step(epoch, global_step)
            t_lengths, t_ac, t_ac5 = seq2seq.policy_gen_eval(test_batches, classifier)
            p.put_list([t_lengths, t_ac, t_ac5])

            if kl_w > 0.5 and t_ac5 > ac5:
                path = join(output_dir, "breakpoints/at_step_%d.ckpt" % global_step)
                save_path = saver0.save(sess0, path)

        global_step += 1

"""RESTORE BEST MODEL"""
path = join(output_dir, "breakpoints/best_test_bleu.ckpt")
saver0.restore(sess0, path)
# TODO: eval train ac

"""GENERATE"""
# TRAIN SET
train_batches = batch_generator(
    train_data, start_i, end_i, batch_size, permutate=False)
(train_recon_loss, train_kl_loss, train_bow_loss,
 perplexity, train_bleu_score, precisions, generation_corpus) = seq2seq.infer_and_eval(train_batches)
write_out(train_out_f, generation_corpus, index2word)
p("BEST TRAIN BLEU: %.1f" % train_bleu_score)

# TEST SET
generation_corpus = seq2seq.infer_and_eval(test_batches)[-1]
write_out(test_out_f, generation_corpus, index2word)

log_f.close()
