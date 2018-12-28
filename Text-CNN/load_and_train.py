# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 17:41:11 2018

@author: kenyon
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import re
import tensorflow as tf
import numpy as np
from text_cnn import textCNN

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load(test_prob):
    pos = []
    neg = []
    labels = []
    
    max_length = 0 #record max sequence length
    neg_file = open('./data/rt-polarity.neg',encoding='utf-8')
    for line in neg_file:
        new_line = clean_str(str.strip(line))
        neg.append(new_line) #str.strip remove space and /n in the start and the end
        labels.append([0,1])
        
        length = len(new_line)
        if(length>max_length):
            max_length = length
        
    neg_file.close()
    
    pos_file = open('./data/rt-polarity.pos',encoding='utf-8')
    for line in pos_file:
        new_line = clean_str(str.strip(line))
        neg.append(new_line) #str.strip remove space and /n in the start and the end
        labels.append([1,0])
        
        length = len(new_line)
        if(length>max_length):
            max_length = length
    pos_file.close()
    
    compile_file = neg + pos
    
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_length)
    vocab_processor.fit(compile_file)
    
    x = np.array(list(vocab_processor.fit_transform(compile_file)))
    y = np.array(labels)
    
    #shuffle data
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)
    
    x_train = x[:int(x.shape[0]*(1-test_prob)),:]
    y_train = y[:int(y.shape[0]*(1-test_prob)),:]
    
    x_test = x[int(x.shape[0]*(1-test_prob)):,:]
    y_test = y[int(y.shape[0]*(1-test_prob)):,:]
    return x_train ,y_train, x_test, y_test, max_length, vocab_processor

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]  #yield 是一个类似 return 的关键字，只是这个函数返回的是个生成器， yield一次就增加一个

if __name__ == '__main__':
    x_train ,y_train, x_test, y_test, max_length, vocab_processor = load(0.01)
    embedding_size = 200

    textCNN = textCNN(max_length, y_train.shape[1], word_number=len(vocab_processor.vocabulary_),
                      embedding_size=embedding_size, kernel_number=10, kernel_size=[2,3,4], dropout_prob=0.4)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(textCNN.losses)
    epochs = 200
    batches = batch_iter(list(zip(x_train, y_train)), batch_size=64, num_epochs=epochs)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        #for i in range(epochs):
        for i,batch in enumerate(batches):
            x_batch, y_batch = zip(*batch)
            feed_train = {textCNN.x : x_batch,
                        textCNN.y : y_batch
                        }
            _,loss_val = sess.run([optimizer,textCNN.losses],feed_dict=feed_train)
            print(loss_val)
            if i%100 == 0:
                feed_test = {textCNN.x : x_test,
                            textCNN.y : y_test
                            }
                acc_val = sess.run([textCNN.accuracy],feed_dict=feed_test)
                print('--------------')
                print('acc: ',acc_val)
                print('--------------')
