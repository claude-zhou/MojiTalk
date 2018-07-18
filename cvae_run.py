import tensorflow as tf
import numpy as np
import json
from time import gmtime, strftime
from os import makedirs, chdir
from os.path import join, dirname

from helpers import *
from cvae import CVAE

"""hyper-params"""
from params.full import *

is_seq2seq = False
if is_seq2seq:
    num_epoch = 6
else:
    num_epoch = 8
test_step = 50
num_gpu = 1
decoder_layer = 1

"""directories"""
chdir('mojitalk_data')
if is_seq2seq:
    output_dir = strftime("seq2seq/%m-%d_%H-%M-%S", gmtime())
else:
    output_dir = strftime("cvae/%m-%d_%H-%M-%S", gmtime())
train_out_f = join(output_dir, "train.out")
test_out_f = join(output_dir, "test.out")  # dev.out
vocab_f = "vocab.ori"
train_ori_f = "train.ori"
train_rep_f = "train.rep"
test_ori_f = "test.ori"  # dev.ori
test_rep_f = "test.rep"  # dev.rep

makedirs(dirname(join(output_dir, "breakpoints/")), exist_ok=True)
log_f = open(join(output_dir, "log.txt"), "w", encoding='utf-8')

"""build vocabulary"""
word2index, index2word = build_vocab(vocab_f)
start_i, end_i = word2index['<s>'], word2index['</s>']
vocab_size = len(word2index)
p = Printer(log_f, index2word)

"""build graph"""
kl_ceiling = 0.48
cvae = CVAE(vocab_size, embed_size, num_unit, latent_dim, emoji_dim, batch_size,
            kl_ceiling, 1, decoder_layer,
            start_i, end_i, beam_width, maximum_iterations, max_gradient_norm, lr, dropout, num_gpu, cell_type,
            is_seq2seq=is_seq2seq)

"""build data"""
train_data = build_data(train_ori_f, train_rep_f, word2index)
test_data = build_data(test_ori_f, test_rep_f, word2index)
test_batches = batch_generator(test_data, start_i, end_i, batch_size, permutate=False)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    """init params"""
    sess.run(tf.global_variables_initializer())
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    saver = tf.train.Saver(var_list=train_vars, max_to_keep=100)
    if not is_seq2seq:
        saver.restore(sess, "seq2seq/07-17_05-49-50/breakpoints/at_step_18000.ckpt")
    cvae.set_sess(sess)

    total_step = (num_epoch * len(train_data[0]) / batch_size)
    global_step = 1
    start_epoch = 1

    for epoch in range(start_epoch, num_epoch + 1):
        train_batches = batch_generator(
            train_data, start_i, end_i, batch_size)

        recon_l = []
        kl_l = []
        bow_l = []
        for batch in train_batches:
            """ TRAIN """
            if is_seq2seq:
                kl_weight = 1.
            else:
                kl_weight = get_kl_weight(global_step, total_step, 0.75)
            recon_loss, kl_loss, bow_loss = cvae.train_update(batch, kl_weight)
            recon_l.append(recon_loss)
            kl_l.append(kl_loss)
            bow_l.append(bow_loss)

            if global_step % test_step == 0:
                time_now = strftime("%m-%d %H:%M:%S", gmtime())
                p.put_step(epoch, global_step)
                p.put_list([np.mean(recon_l), np.mean(kl_l), np.mean(bow_l)])
                recon_l = []
                kl_l = []
                bow_l = []
            if global_step % (test_step * 10) == 0:
                """ EVAL and INFER """
                # TEST
                (test_recon_loss, test_kl_loss, test_bow_loss,
                 test_ppl, test_bleu_score, precisions, _) = cvae.infer_and_eval(test_batches)
                p.put_example(cvae)

                p.put_step(epoch, global_step)
                put_eval(
                    test_recon_loss, test_kl_loss, test_bow_loss,
                    test_ppl, test_bleu_score, precisions, "TEST", log_f)

                if kl_weight >= 0.35:
                    path = join(output_dir, "breakpoints/at_step_%d.ckpt" % global_step)
                    save_path = saver.save(sess, path)
            global_step += 1

    """GENERATE"""
    # TRAIN SET
    train_batches = batch_generator(
        train_data, start_i, end_i, batch_size, permutate=False)
    (train_recon_loss, train_kl_loss, train_bow_loss,
     perplexity, train_bleu_score, precisions, generation_corpus) = cvae.infer_and_eval(train_batches)
    write_out(train_out_f, generation_corpus, index2word)
    p("BEST TRAIN BLEU: %.1f BEST TRAIN PPL: %.3f" % (train_bleu_score, perplexity))

    # TEST SET
    generation_corpus = cvae.infer_and_eval(test_batches)[-1]
    write_out(test_out_f, generation_corpus, index2word)

log_f.close()
