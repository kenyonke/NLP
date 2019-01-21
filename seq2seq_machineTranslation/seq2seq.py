import tensorflow as tf
import numpy as np
from get_data import data

'''
seq2seq model
'''

#load data
data = data()

#encoder
def encoder(xs, source_lens, hidden_size, num_layers, embedding_size):
    #eng_embed = tf.Variable([len(data.eng_word2id.keys()), embedding_size], dtype=tf.float32)
    #seq_embedding = tf.nn.embedding_lookup(eng_embed, xs)
    #print(seq_embedding.shape)
    
    def get_lstmCell(hidden_size): #1.5之后不能再使用[lstm_cell]*num_layers这种方法
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        return lstm_cell
        
    #LSTM model
    mlstm_cell = tf.contrib.rnn.MultiRNNCell([get_lstmCell(hidden_size) for _ in range(num_layers)])
    #init_s = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    seq_embedding = tf.contrib.layers.embed_sequence(xs, len(data.eng_word2id.keys()), embedding_size)
    en_outputs, en_final_states = tf.nn.dynamic_rnn(mlstm_cell,
                                                    seq_embedding,
                                                    source_lens,
                                                    dtype=tf.float32)
    return en_outputs, en_final_states


#decoder training
def decoder_train(en_final_states, seq_embed, sequence_length, max_length, decoder_cell, output_layer):
    
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=seq_embed,
                                                        sequence_length=sequence_length,
                                                        time_major = False,
                                                        name='training_helper')
    training_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,
                                                       helper = training_helper,
                                                       initial_state = en_final_states,
                                                       output_layer = output_layer)
    train_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder = training_decoder,
                                                          impute_finished=True,
                                                          maximum_iterations=max_length)
    return train_outputs

#decoder prediction
def decoder_pre(en_final_states, french_embed, max_length, decoder_cell, batch_size, output_layer):
    s_id = data.fr_word2id['<GO>']
    e_id = data.fr_word2id['<EOS>']
    
    start_tokens = np.array([s_id for _ in range(batch_size)])
    
    pre_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(french_embed,
                                                          start_tokens,
                                                          e_id)
    pre_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                  pre_helper,
                                                  en_final_states,
                                                  output_layer)
    pre_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(pre_decoder,
                                                        impute_finished=True,
                                                        maximum_iterations=max_length)
    return pre_outputs


def decoder(en_final_states,french_input, sequence_length, hidden_size,
            layers_num, embedding_size, batch_size, max_length):
    french_embed = tf.Variable(tf.zeros([len(data.fr_word2id),embedding_size]))
    seq_embdding = tf.nn.embedding_lookup(french_embed,french_input)
    
    def get_lstmCell(hidden_size): #1.5之后不能再使用[lstm_cell]*num_layers这种方法
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        return lstm_cell
    decoder_cell = tf.contrib.rnn.MultiRNNCell([get_lstmCell(hidden_size) for _ in range(layers_num)])
    
    output_layer = tf.layers.Dense(len(data.fr_word2id))
    
    #踩过的坑：training decoder中输入的是句子的embedding， prediction decoder中输入的是整个 embedding matrix
    with tf.variable_scope('decoder'):
        train_outputs = decoder_train(en_final_states,
                                      seq_embdding,
                                      sequence_length,
                                      max_length,
                                      decoder_cell,
                                      output_layer)
    
    with tf.variable_scope('decoder', reuse=True):
        pre_outputs = decoder_pre(en_final_states,
                                  french_embed,
                                  max_length,
                                  decoder_cell,
                                  batch_size,
                                  output_layer)
    
    return train_outputs, pre_outputs
    
def seq2seq_eng2fr(xs,
                   ys,
                   source_lens,
                   target_lens,
                   hidden_size,
                   num_layers,
                   embedding_size,
                   batch_size,
                   max_length=25):
    
    _,en_final_states = encoder(xs, source_lens, hidden_size, num_layers, embedding_size)
    
    train_outputs,pre_outputs = decoder(en_final_states,
                                        ys,
                                        target_lens,
                                        hidden_size,
                                        num_layers,
                                        embedding_size,
                                        batch_size,
                                        max_length=max_length)
    
    return train_outputs,pre_outputs

def train_model(train_outputs, targets, target_sequence_len, max_target_sequence_len, learning_rate):
    masks = tf.sequence_mask(target_sequence_len, max_target_sequence_len, dtype=tf.float32, name='masks')
    loss = tf.contrib.seq2seq.sequence_loss(tf.identity(train_outputs.rnn_output), targets, masks)
    #opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)   
    gradients = optimizer.compute_gradients(loss)
    clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    opt = optimizer.apply_gradients(clipped_gradients)
    return loss, opt


if __name__ == '__main__':
    source_len = 20
    target_len = 25
    sources = tf.placeholder(tf.int32,[None,source_len])
    targets = tf.placeholder(tf.int32,[None,target_len])
    source_lens = tf.placeholder(tf.int32,[None,])
    target_lens = tf.placeholder(tf.int32,[None,])
    batch_size = 100
    learning_rate = 0.001
    max_target_sequence_len = 25
    epoches = 5
    
    #add <'GO'> before French sequences in training
    #修改了很久的错误：decoder中的输入target是加了<'GO'>的，但是训练时的target是不需要加上<'GO'>的原始句子！！！
    ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    changed_targets = tf.concat([tf.fill([batch_size, 1], data.fr_word2id["<GO>"]), ending], 1)
    
    test = 'i dislike grapefruit , lemons , and peaches .'
    test_int = data.seq2id(test,source_len)
    
    train_outputs, pre_outputs =seq2seq_eng2fr(tf.reverse(sources, [-1]),#sources,
                                               changed_targets,
                                               source_lens,
                                               target_lens,
                                               hidden_size=128,
                                               num_layers=1,
                                               embedding_size=100,
                                               batch_size=batch_size,
                                               max_length = max_target_sequence_len)
    #pre_outputs.sample_id
    loss,optimizer = train_model(train_outputs, targets, target_lens, max_target_sequence_len, learning_rate)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoches):
            for source_batch, source_lengths, target_batch, target_lengths in data.get_batches(batch_size):
                feed_dict = {sources : source_batch,
                             targets : target_batch,
                             source_lens : source_lengths,
                             target_lens : target_lengths}
                loss_val, _ = sess.run([loss,optimizer],feed_dict)
                print(loss_val)
                
            
        prediction = sess.run([pre_outputs.sample_id],feed_dict={sources : [test_int]*batch_size,
                                                       source_lens : [source_len]*batch_size,
                                                       target_lens : [target_len]*batch_size})[0][0]
        print(prediction)
        print(data.id2seq(prediction))