# -*- coding: utf-8 -*-
from collections import Counter
from sklearn.metrics import f1_score
import random
import re
import copy
import sys

class CommandLine:
    def __init__(self):
        if (sys.argv[1] == ''):
            print("please type trainfile's path and testfile's path")
            sys.exit()
        else:
            self.trainpath = sys.argv[1]
            self.testpath = sys.argv[2]
            
class doc:
    def __init__(self,trainpath,testpath):
        self.trainpath = trainpath
        self.testpath = testpath
        
        self.word_tag_features = doc.features(self.trainpath,doc.word_tag)
        self.W_t = self.word_tag_train()
        print('micro-F1: ',self.f1(self.W_t,'word_tag'))
        print('')
        
        self.tag_tag_features = doc.features(self.trainpath,doc.word_tag_tag)
        self.W_tt = self.word_tag_tag_train()
        print('micro-F1: ',self.f1(self.W_tt,'tag_tag'))
        print('')
        
        self.preword_word_tag_tag_tag_features = doc.features(self.trainpath,doc.pw_w_ppt_pt_t)
        self.W_pttt = self.preword_word_tag_tag_tag_train()
        print('micro-F1: ',self.f1(self.W_pttt,'preword_word_tag_tag_tag'))
        print('')
        
    #retrieve features based on the input feature type for a document, then save as a list   
    def features(trainpath,features_function):
        document = []
        random.seed(9)
        with open(trainpath) as f:
            for line in f:
                line_features = features_function(line)
                document.append(line_features)
        random.shuffle(document)
        return document

    #a function that retrieve words and their tags for a line and put them into a list
    def word_tag(line):
        new_line = re.sub("[\s]"," ",line).split()
        length = len(new_line)
        words = new_line[:int(length/2)]
        pos = new_line[int(length/2):]
        line_features = []
        for i in range(len(words)):
            line_features.append((words[i],pos[i]))
        return line_features
    
    def word_tag_train(self):
        print('current word with current label')
        #W = {(word,tag):freq,......}
        W = Counter()
        W_store = []
        iteration = 10
        for k in range(iteration):
            for i in range(len(self.word_tag_features)):
                word_feature = Counter(self.word_tag_features[i])
                
                line = [] #line list stores current words
                tags = [] #tags list stores current tags
                for j in range(len(self.word_tag_features[i])):
                    line.append(self.word_tag_features[i][j][0])
                    tags.append(self.word_tag_features[i][j][1])
                
                predict_tags = doc.line_beam_search(line,W,'word_tag')
                predict_word_feature = []
                #print(line)
                #print(predict_tags)
                for index in range(len(line)):
                    predict_word_feature.append((line[index],predict_tags[index]))
                predict_word_feature = Counter(predict_word_feature)
                if predict_tags != tags:
                    W.update(word_feature)
                    W.subtract(predict_word_feature)
                                                
            #store every iteration's W for average  
            W_store.append(copy.deepcopy(W))
        #average W in W_store
        for key in W.keys():
            sum_word = 0
            for i in range(iteration):
                sum_word += W_store[i][key]
            W[key] = sum_word/iteration
        
        #classify different classes and sort them
        W_O = {}
        W_LOC = {}
        W_MISC = {}
        W_PER = {}
        W_ORG = {}
        for words_tags,score in W.items():
            if words_tags[1] == 'O':
                W_O[words_tags] = score
            if words_tags[1] == 'LOC':
                W_LOC[words_tags] = score
            if words_tags[1] == 'MISC':
                W_MISC[words_tags] = score
            if words_tags[1] == 'ORG':
                W_ORG[words_tags] = score
            if words_tags[1] == 'PER':
                W_PER[words_tags] = score
        print('O: ',sorted(W_O.items(), key = lambda item:item[1],reverse=True)[:10])
        print('ORG: ',sorted(W_ORG.items(), key = lambda item:item[1],reverse=True)[:10])
        print('LOC: ',sorted(W_LOC.items(), key = lambda item:item[1],reverse=True)[:10])
        print('PER: ',sorted(W_PER.items(), key = lambda item:item[1],reverse=True)[:10])
        print('MISC: ',sorted(W_MISC.items(), key = lambda item:item[1],reverse=True)[:10])
        return W

    #a function that retrieve words and their previous tags and current tags and put them in a list for one line in trainning text
    def word_tag_tag(line):
        new_line = re.sub("[\s]"," ",line).split()
        length = len(new_line)
        words = new_line[:int(length/2)]
        pos = new_line[int(length/2):]
        line_features = []
        for i in range(len(words)):
            if i==0:
                line_features.append((words[i],('None',pos[i])))
            else:
                line_features.append((words[i],(pos[i-1],pos[i])))
        return line_features
    
    def word_tag_tag_train(self):
        print('current word with previous label and current label')
        #W = {(word,(tag_previous,tag_current)):freq,......}
        W=Counter()
        W_store = []
        iteration = 10
        for k in range(iteration):
            for i in range(len(self.tag_tag_features)):
                word_feature = Counter(self.tag_tag_features[i])
                
                line = [] #line list stores current words
                tags = [] #tags list stores current tags
                for j in range(len(self.tag_tag_features[i])):
                    line.append(self.tag_tag_features[i][j][0])
                    tags.append(self.tag_tag_features[i][j][1][1])
                
                predict_tags = doc.line_beam_search(line,W,'tag_tag')
                predict_word_feature = []
                for index in range(len(line)):
                    if index == 0:
                        predict_word_feature.append((line[index],('None',predict_tags[index])))
                    else:
                        predict_word_feature.append((line[index],(predict_tags[index-1],predict_tags[index])))
                predict_word_feature = Counter(predict_word_feature)
                if predict_tags != tags:
                    W.update(word_feature)
                    W.subtract(predict_word_feature)

            #store every iteration's W for average  
            W_store.append(copy.deepcopy(W))
        #average W in W_store
        for key in W.keys():
            sum_word = 0
            for i in range(iteration):
                sum_word += W_store[i][key]
            W[key] = sum_word/iteration            
        
        #classify different classes and sort them
        W_O = {}
        W_LOC = {}
        W_MISC = {}
        W_PER = {}
        W_ORG = {}
        for words_tags,score in W.items():
            if words_tags[1][1] == 'O':
                W_O[words_tags] = score
            if words_tags[1][1] == 'LOC':
                W_LOC[words_tags] = score
            if words_tags[1][1] == 'MISC':
                W_MISC[words_tags] = score
            if words_tags[1][1] == 'ORG':
                W_ORG[words_tags] = score
            if words_tags[1][1] == 'PER':
                W_PER[words_tags] = score
        print('O: ',sorted(W_O.items(), key = lambda item:item[1],reverse=True)[:10])
        print('ORG: ',sorted(W_ORG.items(), key = lambda item:item[1],reverse=True)[:10])
        print('LOC: ',sorted(W_LOC.items(), key = lambda item:item[1],reverse=True)[:10])
        print('PER: ',sorted(W_PER.items(), key = lambda item:item[1],reverse=True)[:10])
        print('MISC: ',sorted(W_MISC.items(), key = lambda item:item[1],reverse=True)[:10])                
        return W

    #a function that retrieves bigram words and trigram tags for a line and put them into a list
    def pw_w_ppt_pt_t(line):
        new_line = re.sub("[\s]"," ",line).split()
        length = len(new_line)
        words = new_line[:int(length/2)]
        pos = new_line[int(length/2):]
        line_features = []
        for i in range(len(words)):
            if i==0:
                line_features.append((('None',words[i]),('None','None',pos[i])))
            elif i==1:
                line_features.append(((words[i-1],words[i]),('None',pos[i-1],pos[i])))
            else:
                line_features.append(((words[i-1],words[i]),(pos[i-2],pos[i-1],pos[i])))
        return line_features
    
    def preword_word_tag_tag_tag_train(self):
        print('bigram words with trigram labels')
        #W = {((preword,word),(tag_pre-previous,tag_previous,tag_current)):freq,......}
        W=Counter()
        W_store = []
        iteration = 10
        for k in range(iteration):
            for i in range(len(self.preword_word_tag_tag_tag_features)):
                word_feature = Counter(self.preword_word_tag_tag_tag_features[i])
                
                line = [] #line list stores sets (preword,word)
                tags = [] #tags list stores current tags
                for j in range(len(self.preword_word_tag_tag_tag_features[i])):
                    line.append(self.preword_word_tag_tag_tag_features[i][j][0])
                    tags.append(self.preword_word_tag_tag_tag_features[i][j][1][2])
                
                predict_tags = doc.line_beam_search(line,W,'preword_word_tag_tag_tag')
                predict_word_feature = []
                
                #count predicted features
                for index in range(len(line)):
                    if index == 0:
                        predict_word_feature.append((line[index],('None','None',predict_tags[index])))
                    elif index == 1:
                        predict_word_feature.append((line[index],('None',predict_tags[index-1],predict_tags[index])))
                    else:
                        predict_word_feature.append((line[index],(predict_tags[index-2],predict_tags[index-1],predict_tags[index])))
                predict_word_feature = Counter(predict_word_feature)
                
                #update W
                if predict_tags != tags:
                    W.update(word_feature)
                    W.subtract(predict_word_feature)
            #store every iteration's W for average  
            W_store.append(copy.deepcopy(W))
        #average W in W_store
        for key in W.keys():
            sum_word = 0
            for i in range(iteration):
                sum_word += W_store[i][key]
            W[key] = sum_word/iteration
        
        #classify different classes and sort them
        W_O = {}
        W_LOC = {}
        W_MISC = {}
        W_PER = {}
        W_ORG = {}
        for words_tags,score in W.items():
            if words_tags[1][2] == 'O':
                W_O[words_tags] = score
            if words_tags[1][2] == 'LOC':
                W_LOC[words_tags] = score
            if words_tags[1][2] == 'MISC':
                W_MISC[words_tags] = score
            if words_tags[1][2] == 'ORG':
                W_ORG[words_tags] = score
            if words_tags[1][2] == 'PER':
                W_PER[words_tags] = score
        print('O: ',sorted(W_O.items(), key = lambda item:item[1],reverse=True)[:10])
        print('ORG: ',sorted(W_ORG.items(), key = lambda item:item[1],reverse=True)[:10])
        print('LOC: ',sorted(W_LOC.items(), key = lambda item:item[1],reverse=True)[:10])
        print('PER: ',sorted(W_PER.items(), key = lambda item:item[1],reverse=True)[:10])
        print('MISC: ',sorted(W_MISC.items(), key = lambda item:item[1],reverse=True)[:10])
        
        return W
    
    def line_beam_search(sentence,W,feature_type):
        k = 25
        tag_set = ('O','ORG','LOC','MISC','PER')
        # word_tag features' beam search
        if feature_type == 'word_tag':
            beam = [[['None'],0]]
            for i in range(len(sentence)):
                beam_n = []
                for b in beam: #At first, b[0] = ['None'],which contains tags orderly
                    for tag in tag_set:
                        if (sentence[i],tag) not in W.keys():
                            score = 0
                        else:
                            score = W[(sentence[i],tag)]
                        new_list = b[0][:]
                        new_list.append(tag)
                        beam_n.append([new_list,b[1]+score])
                    beam = sorted(beam_n, key=lambda x:x[1],reverse=True)[:k]
            result = beam[0][0][1:]
        
        if feature_type == 'tag_tag':
            beam = [[['None'],0]]
            for i in range(len(sentence)):
                beam_n = []
                for b in beam: #At first, b[0] = ['None'],which contains tags orderly
                    for tag in tag_set:
                        if (sentence[i],(b[0][i],tag)) not in W.keys():
                            score = 0
                        else:
                            score = W[sentence[i],(b[0][i],tag)]
                        new_list = b[0][:]
                        new_list.append(tag)
                        beam_n.append([new_list,b[1]+score])
                    beam = sorted(beam_n, key=lambda x:x[1],reverse=True)[:k]
            result = beam[0][0][1:]
                    
        if feature_type == 'preword_word_tag_tag_tag':                         
            beam = [[['None','None'],0]]
            for i in range(len(sentence)):
                beam_n = []
                for b in beam: #At first, b[0] = ['None','None'],which contains tags orderly
                    for tag in tag_set:
                        if (sentence[i],(b[0][i],b[0][i+1],tag)) not in W.keys():
                            score = 0
                        else:
                            score = W[(sentence[i],(b[0][i],b[0][i+1],tag))]
                        new_list = b[0][:]
                        new_list.append(tag)
                        beam_n.append([new_list,b[1]+score])
                    beam = sorted(beam_n, key=lambda x:x[1],reverse=True)[:k]
            result = beam[0][0][2:]
        return result

    def f1(self,W,feature_type):
        predict = []
        correct = []
        with open(self.testpath) as f:
            for line in f:
                predict_labels = []
                new_line = re.sub("[\s]"," ",line).split()
                length = len(new_line)
                sentence = new_line[:int(length/2)]
                labels = new_line[int(length/2):]
                if feature_type == 'preword_word_tag_tag_tag':
                    preword_word_sentence = []
                    for i in range(len(sentence)):
                        if i==0:
                            preword_word_sentence.append(('None',sentence[i]))
                        else:
                            preword_word_sentence.append((sentence[i-1],sentence[i]))                    
                    predict_labels = doc.line_beam_search(preword_word_sentence,W,'preword_word_tag_tag_tag')
                else:
                    predict_labels = doc.line_beam_search(sentence,W,feature_type)
                for i in range(len(predict_labels)):
                    predict.append(predict_labels[i])
                    correct.append(labels[i])
        f1_micro = f1_score(correct, predict, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])    
        return f1_micro

if __name__ == '__main__':
    config = CommandLine()
    doc(config.trainpath,config.testpath)