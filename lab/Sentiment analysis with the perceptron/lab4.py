# -*- coding: utf-8 -*-
import os
import re
import random
import copy
import sys
import nltk
import math
import matplotlib.pyplot as plt
from collections import Counter

class CommandLine:
    def __init__(self):
        if (sys.argv[1] == ''):
            print('please type file path')
            sys.exit()
        else:
            self.filepath = sys.argv[1]

class model:
    def __init__(self,filepath):
       self.document = filepath
       self.training()
       
    def training(self):
        #all of the feathers of words have the same format as: [({d1:count},y1),({d2:count},y2),...]
        #bag-of-words
        bags=[]
        #bigram
        bigrams=[]
        #tfidf
        tfidfs = []
        #d_sum is the number of total documents
        d_sum = 0
        #start calculating bags[], bigrams[] and tfidfs[]
        neg_train_path = self.document + '/txt_sentoken/neg/train/'
        files= os.listdir(neg_train_path)
        for file in files:
            d_sum += 1
            unigram = []
            bigram = []
            with open(neg_train_path+'/'+file,'r') as f:
                for line in f:
                    sentence = re.sub("[^\w']"," ",line).split()
                    unigram.extend(sentence)
                    bigram.extend(nltk.bigrams(sentence, pad_left=True, pad_right=True))
            bags.append((Counter(unigram),-1))
            bigrams.append((Counter(bigram),-1))
        
        pos_train_path = self.document + '/txt_sentoken/pos/train/'
        files= os.listdir(pos_train_path)
        for file in files:
            d_sum += 1
            unigram = []
            bigram = []
            with open(pos_train_path+'/'+file,'r') as f:
                for line in f:
                    sentence = re.sub("[^\w']"," ",line).split()
                    unigram.extend(sentence)
                    bigram.extend(nltk.bigrams(sentence, pad_left=True, pad_right=True))
            bags.append((Counter(unigram),1))
            bigrams.append((Counter(bigram),1))
        
        df_w = {}
        for (tf_d,y) in bags:
            for (word,freq) in tf_d.items():
                if word not in df_w.keys():
                    df_w[word] = 0
                    for new_tf_d,y in bags:
                        if word in new_tf_d.keys():
                            df_w[word] += 1
        for (tf_d,y) in bags:
            tfidf = {}
            for(word,freq) in tf_d.items():
                 tfidf[word] = 0
                 tfidf[word] = freq * math.log(d_sum/df_w[word])
            tfidfs.append((tfidf,y))
        
        
        #start traing W
        #fix random seed in order to make result reproducible
        random.seed(100)
        #W is the model we need to train
        W = {}
        #W_store can store differnt iterative model
        W_store = []        
        #times of iteration
        maxIter = 10
        plt.close()
        plt.ion()
        plt.title('unigram')
        for i in range(maxIter): 
            random.shuffle(bags)
            for (x,y) in bags:
                #initail predictive value of y
                y_predict = 0
                for (word, count) in x.items():
                    if word not in W.keys():
                        W[word] = 0
                    #y^ = sgn(W(x) * phi(x))
                    y_predict += count * W[word]
                #update W
                if y == -1:
                    if y_predict >= 0:
                        for (word,count) in x.items():
                            W[word] = W[word] + y * count
                else:
                    if y_predict <= 0:
                        for (word,count) in x.items():
                            W[word] = W[word] + y * count
            W_store.append(copy.deepcopy(W))
            acc = self.validation(W)
            plt.bar(i+1,acc)
            plt.text(i+1, acc, acc, ha='center',va= 'bottom',fontsize=7)
            plt.pause(0.2)
        #average all of the updated W in W_store
        for word in W.keys():
            sum_word = 0
            for i in range(maxIter):
                sum_word += W_store[i][word]
            W[word] = sum_word/maxIter
        print('-----unigram-----')
        print('Accuracy of average weight vectors',self.validation(W))
        plt.pause(3)
        plt.close()
        
        plt.title('bigram')
        print("-----bigram-----")
        bigram_W = {}
        bigram_W_Store = []
        for i in range(maxIter): 
            random.shuffle(bigrams)
            for (x,y) in bigrams:
                y_predict = 0
                for (pair, count) in x.items():
                    if pair not in bigram_W.keys():
                        bigram_W[pair] = 0
                    #y^ = sgn(bigram_W(x) * phi(x))
                    y_predict += count * bigram_W[pair]
                #update bigram_W
                if y == -1:
                    if y_predict >= 0:
                        for (pair,count) in x.items():
                            bigram_W[pair] = bigram_W[pair] + y * count
                else:
                    if y_predict <= 0:
                        for (pair,count) in x.items():
                            bigram_W[pair] = bigram_W[pair] + y * count
            bigram_W_Store.append(copy.deepcopy(bigram_W))
            acc = self.validation_bigram(bigram_W)
            plt.bar(i+1,acc)
            plt.text(i+1, acc, acc,ha='center',va= 'bottom',fontsize=7)
            plt.pause(0.01)
        #average all of the updated bigram_W in bigram_W_Store
        for pair in bigram_W.keys():
            sum_pair = 0
            for i in range(maxIter):
                sum_pair += bigram_W_Store[i][pair]
            bigram_W[pair] = sum_pair/maxIter
        print('Accuracy of average weight vectors',self.validation_bigram(bigram_W))
        plt.pause(3)
        plt.close()
        
        plt.title('tfidf')
        print('-----tfidf-----')
        tfidf_W = {}
        #W_store can store differnt iterative model
        tfidf_W_store = []        
        #times of iteration
        for i in range(maxIter): 
            random.shuffle(tfidfs)
            for (x,y) in tfidfs:
                #initail predictive value of y
                y_predict = 0
                for (word, count) in x.items():
                    if word not in tfidf_W.keys():
                        tfidf_W[word] = 0
                    #y^ = sgn(W(x) * phi(x))
                    y_predict += count * tfidf_W[word]
                #update W
                if y == -1:
                    if y_predict >= 0:
                        for (word,count) in x.items():
                            tfidf_W[word] = tfidf_W[word] + y * count
                else:
                    if y_predict <= 0:
                        for (word,count) in x.items():
                            tfidf_W[word] = tfidf_W[word] + y * count
            tfidf_W_store.append(copy.deepcopy(tfidf_W))
            acc = self.validation_tfidf(tfidf_W,df_w,d_sum)
            plt.bar(i+1,acc)
            plt.text(i+1, acc, acc,ha='center',va= 'bottom',fontsize=7)
            plt.pause(0.2)
        #average all of the updated W in W_store
        for word in tfidf_W.keys():
            sum_word = 0
            for i in range(maxIter):
                sum_word += tfidf_W_store[i][word]
            tfidf_W[word] = sum_word/maxIter
        plt.pause(3)
        print('Accuracy of average weight vectors',self.validation_tfidf(tfidf_W,df_w,d_sum))    
        
    def validation(self,W):
        pos_test_path = self.document + '/txt_sentoken/pos/test/'
        files= os.listdir(pos_test_path)
        pos = 0
        total = 0
        for file in files:
            total += 1
            result = 0
            with open(pos_test_path+'/'+file,'r') as f:
                for line in f:
                    sentence = re.sub("[^\w']"," ",line).split()
                    for word in sentence:
                        if word in W.keys():
                            result += W[word]
            if result >0:
                pos += 1

        neg_test_path = self.document + '/txt_sentoken/neg/test/'
        files= os.listdir(neg_test_path)
        neg = 0
        for file in files:
            total += 1
            result = 0
            with open(neg_test_path+'/'+file,'r') as f:    
                for line in f:
                    sentence = re.sub("[^\w']"," ",line).split()
                    for word in sentence:
                        if word in W.keys():
                            result += W[word]
            if result < 0:
                neg += 1
        return (pos+neg)/total

    def validation_bigram(self,W_bigram):
        pos_test_path = self.document + '/txt_sentoken/pos/test/'
        files= os.listdir(pos_test_path)
        pos = 0
        total = 0
        neg = 0
        for file in files:
            total += 1
            d = []
            result = 0
            with open(pos_test_path+'/'+file,'r') as f:
                for line in f:
                    sentence = re.sub("[^\w']"," ",line).split()
                    d.extend(nltk.bigrams(sentence, pad_left=True, pad_right=True))
                for pair in d:
                    if pair in W_bigram.keys():
                        result += W_bigram[pair]
            if result >0:
                pos += 1
        
        neg_test_path = self.document + '/txt_sentoken/neg/test/'
        files= os.listdir(neg_test_path)
        for file in files:
            total += 1
            d = []
            result = 0
            with open(neg_test_path+'/'+file,'r') as f:
                for line in f:
                    sentence = re.sub("[^\w']"," ",line).split()
                    d.extend(nltk.bigrams(sentence, pad_left=True, pad_right=True))
                for pair in d:
                    if pair in W_bigram.keys():
                        result += W_bigram[pair]
            if result <0:
                neg += 1
        return (pos+neg)/total
        
    def validation_tfidf(self,W,original_df_w,d_sum):
        df_w = copy.deepcopy(original_df_w)
        pos_test_path = self.document + '/txt_sentoken/pos/test/'
        files= os.listdir(pos_test_path)
        pos = 0
        total = 0
        for file in files:
            # d is used for storing all of the splited words in a document
            d = []
            total += 1
            with open(pos_test_path+'/'+file,'r') as f:
                result = 0
                #d_tfidf is used to store values of tfidf of all the words in the document 
                d_tfidf = {}
                for line in f:        
                    sentence = re.sub("[^\w']"," ",line).split()
                    d.extend(sentence)
                #convert tf into tfidf
                for (word,freq) in Counter(d).items():
                    #initial the tfidf value of a word
                    d_tfidf[word] = 0
                    if word not in df_w.keys():
                        df_w[word] = 1
                    d_tfidf[word] = math.log(d_sum/df_w[word])* freq
                #calculate the weighted sum of the document 
                for (word,count) in d_tfidf.items():
                    if word in W.keys():
                        result += W[word] * count
            if result >0:
                pos += 1            
        
        neg_test_path = self.document + '/txt_sentoken/neg/test/'
        files= os.listdir(neg_test_path)
        neg = 0
        for file in files:
            # d is used for storing all of the splited words in a document
            d = []
            total += 1
            with open(neg_test_path+'/'+file,'r') as f:
                result = 0
                #d_tfidf is used to store values of tfidf of all the words in the document 
                d_tfidf = {}
                for line in f:        
                    sentence = re.sub("[^\w']"," ",line).split()
                    d.extend(sentence)
                #convert tf into tfidf
                for (word,freq) in Counter(d).items():
                    #initial the tfidf value of a word
                    d_tfidf[word] = 0
                    if word not in df_w.keys():
                        df_w[word] = 1
                    d_tfidf[word] = math.log(d_sum/df_w[word])* freq
                #calculate the weighted sum of the document 
                for (word,count) in d_tfidf.items():
                    if word in W.keys():
                        result += W[word] * count
            if result < 0:
                neg += 1            
        return (pos+neg)/total
        
if __name__ == '__main__':
    config = CommandLine()
    model = model(config.filepath)
