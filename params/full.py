import tensorflow as tf

"""hyper params for building the graph"""
batch_size = 128    #
num_unit = 128      # num_unit should be equal to embed_size
embed_size = 128    #
latent_dim = 268    #
num_gpu = 1
emoji_dim = 12

# default
lr = 1e-3
max_gradient_norm = 5
maximum_iterations = 50
beam_width = 0
dropout = 0.2
decoder_layer = 1

# GRUCell won't have multiple kinds of state. Wouldn't have to flatten its state before concatenation
cell_type = tf.nn.rnn_cell.GRUCell

"""hyper params for running the graph"""
# num_epoch = 12      #
# test_step = 100     #
