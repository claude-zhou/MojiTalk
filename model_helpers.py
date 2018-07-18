import tensorflow as tf

xavier = tf.contrib.layers.xavier_initializer()

def build_bidirectional_rnn(
        num_dim, inputs, sequence_length, cell_type, num_gpu, drop, base_gpu=0,  dtype=tf.float32):
    # TODO: move rnn cell creation functions to a separate file
    # Construct forward and backward cells
    fw_cell = create_rnn_cell(
        num_dim, base_gpu, cell_type, num_gpu, drop)
    bw_cell = create_rnn_cell(
        num_dim, (base_gpu + 1), cell_type, num_gpu, drop)

    bi_output, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=True)

    return bi_output, bi_state

def create_rnn_cell(num_dim, base_gpu, cell_type, num_gpu, drop):
    # dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
    device_str = "/gpu:%d" % (base_gpu % num_gpu)
    print(device_str)
    single_cell = cell_type(num_dim)
    single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
    single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - drop))
    return single_cell

class Embedding(object):
    def __init__(self, vocab_size, embed_size):
        # TODO: init from embedding
        self.coder = tf.Variable(
            tf.random_normal([vocab_size, embed_size], - 0.5 / embed_size, 0.5 / embed_size),
            name='word_embedding',
            dtype=tf.float32)

    def __call__(self, texts):
        return tf.nn.embedding_lookup(self.coder, texts)
