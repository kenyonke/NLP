# -*- coding: utf-8 -*-
import spacy
import csv
import time
import sklearn
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import cmudict
from nltk.corpus import wordnet
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from spacy.symbols import ORTH, LEMMA , POS, IS_ALPHA

class features():
    def __init__(self,trainset,language):
        self.language = language
        self.eng_dic = Counter()
        self.spa_dic = Counter()
        self.eng_dic = cmudict.dict() #the Carnegie Mellon Pronouncing Dictionary
        with open('synonyms_synonyms_count_dict.pkl','rb') as loadModel:
            self.spa_dic = pickle.load(loadModel)
        self.unigram_freq = Counter() #frequencies of unigram in the corpus 
        self.fivegram_freq = Counter()  #frequencies of n-gram in the corpus 
        self.fourgram_freq = Counter()  #frequencies of n-gram in the corpus 
        self.trigram_freq = Counter()  #frequencies of n-gram in the corpus 
        self.bigram_freq = Counter()  #frequencies of n-gram in the corpus 

        self.get_frequencies(trainset)

    def get_frequencies(self,trainset): 
        with open(trainset,encoding='utf-8', mode = 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            line = ''
            for row in reader:
                #---freqency, n_garam frequency, number of poses a word has---#
                if line != row[1]:                    
                    line = row[1]             
                    words = []
                    bigram = []
                    trigram = []
                    fourgram = []
                    fivegram = []                    
                    for token in nlp(line):
                        #reomve punctuation
                        if token.is_alpha:
                            words.append(token.lemma_.lower())
                            
                    bigram = ngrams(words, 2, pad_left=False, pad_right=False)
                    trigram = ngrams(words, 3, pad_left=False, pad_right=False)
                    fourgram = ngrams(words, 4, pad_left=False, pad_right=False)
                    fivegram = ngrams(words, 5, pad_left=False, pad_right=False)
                                                                    
                    #update n-grams and unigram after a sentence
                    self.unigram_freq.update(Counter(words))
                    self.bigram_freq.update(Counter(bigram))
                    self.trigram_freq.update(Counter(trigram))
                    self.fourgram_freq.update(Counter(fourgram))
                    self.fivegram_freq.update(Counter(fivegram))
        
    #ref:https://stackoverflow.com/questions/46759492/syllable-count-in-python?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa:
    #extract number of syllables of a word
    def syllable_synonyms_count(self,word):
        
        if self.language == 'english':
            #if word in dictionary
            if word in self.eng_dic.keys():
                for pronunciations_list in self.eng_dic[word.lower()]:
                    syllable_count = 0
                    for pronunciation in pronunciations_list:
                        if pronunciation[-1].isdigit():
                            syllable_count += 1
                    
            #if word not in dictionary 
            else:
                word = word.lower()
                syllable_count = 0
                vowels = "aeiouy"
                if word[0] in vowels:
                    syllable_count += 1
                for index in range(1, len(word)):
                    if word[index] in vowels and word[index - 1] not in vowels:
                        syllable_count += 1
                        if word.endswith("e"):
                            syllable_count -= 1
                        if word.endswith('le'):
                            syllable_count += 1
                if syllable_count == 0:
                    syllable_count += 1
                
            synonyms = []
            for syn in wordnet.synsets(word):
                for syn_word in syn.lemmas():
                    synonyms.append(syn_word.name())
            return syllable_count,len(set(synonyms))
        
        elif self.language == 'spanish':
            #dictionary store number of synonym and syllable of a word search in Internet.
            if word not in self.spa_dic.keys():
                synonyms_count = 0
                syllable_count = 0                
            else:
                synonyms_count = self.spa_dic[word][1]
                syllable_count = self.spa_dic[word][0]
            return syllable_count, synonyms_count       
            
    def extract_features(self,words):
        
        if len(words) == 1:
            word = words[0].lower()
            length = len(word)
            freq = self.unigram_freq[word]
            syllable_count,synonyms_count = self.syllable_synonyms_count(word) 
        
        else:
            length = 0
            syllable_count = 0
            synonyms_count = 0
            
            for word in words:
                word = word.lower()
                length += len(word)
                a,b = self.syllable_synonyms_count(word)
                syllable_count += a
                synonyms_count += b
                
            if len(words) == 2:
                freq = self.bigram_freq[ngrams(words, 2, pad_left=False, pad_right=False)]
            elif len(words) == 3:
                freq = self.trigram_freq[ngrams(words, 3, pad_left=False, pad_right=False)]
            elif len(words) == 4:
                freq = self.fourgram_freq[ngrams(words, 4, pad_left=False, pad_right=False)]
            elif len(words) == 5:
                freq = self.fivegram_freq[ngrams(words, 5, pad_left=False, pad_right=False)]
            else:
                freq = 0
            
        return length, freq, syllable_count, synonyms_count #length/5.3
        
            


class Eng_train():
    def __init__(self,model,trainset):
        self.clf = svm.SVC(kernel='rbf') #svm.SVC(kernel='linear')
        self.model = model
        self.features = None
        self.labels = None
        self.set_fea_labels(trainset)
        self.predicted_labels = []
        self.gold_labels = []
    
    def train(self):
        #cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        #title = 'Learning Curves'
        #estimator = svm.SVC(kernel='rbf')
        #self.plot_learning_curve(estimator, title, self.features, self.labels,(0.7, 1.01), cv=cv, n_jobs=4)
        #plt.show()
        self.clf.fit(self.features, self.labels) 
        
    def plot_learning_curve(self,estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    
        plt.legend(loc="best")
        return plt  
      
        
    def set_fea_labels(self,trainset):
        pos_set = ('PUNCT','DET', 'ADJ', 'NOUN', 'CCONJ', 'VERB', 'PART', 'ADP', 'ADV', 'PROPN', 'PRON', 'NUM', 'INTJ', 'X', 'SYM')
        features = []
        labels = []
        with open(trainset,encoding='utf-8', mode = 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            line = ''
            for row in reader:
                if line != row[1]:
                    line = row[1]
                    sentence = []
                    lemma_sentence = []
                    pos_list = []
                    for word in nlp(row[1]):
                        if word.is_alpha:
                            sentence.append(word.text)
                            lemma_sentence.append(word.lemma_)
                            pos_list.append(word.pos_)
                
                label = row[9]
                ori_words = row[4]
                target_words = nlp(ori_words)

                feature = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]             
                if len(target_words) == 1:
                    word = ori_words
                    try:
                        index = sentence.index(word)
                    except:
                        if word=='madness':
                            index = lemma_sentence.index(word)
                        elif word=='sv':
                            index = lemma_sentence.index(word)
                        else:
                            continue
                    pos = pos_list[index]
                    pos_index = pos_set.index(pos)
                    length,freq,syllable,synonyms = self.model.extract_features([lemma_sentence[index]])
                    
                    feature[pos_index] = 1
                    feature.append(length)
                    feature.append(freq)
                    feature.append(syllable)
                    feature.append(synonyms)
                    
                    features.append(feature)
                    labels.append(label)
                    
                else:
                    words = []
                    for word in target_words:
                        if word.is_alpha:
                            try:
                                index = sentence.index(word.text)
                            except:
                                if word.text =='madness':
                                    index = lemma_sentence.index(word.text)
                                elif word.text =='sv':
                                    index = lemma_sentence.index(word.text)
                                else:
                                    continue
                            pos = pos_list[index]
                            pos_index = pos_set.index(pos)
                            feature[pos_index] = 1
                            words.append(lemma_sentence[index])
                    length,freq,syllable,synonyms = self.model.extract_features(words)
                    feature.append(length)
                    feature.append(freq)
                    feature.append(syllable)
                    feature.append(synonyms)
                    features.append(feature)
                    labels.append(label)    
            self.features = features
            self.labels = labels                
        
    def dev(self,testset):
        pos_set = ('PUNCT','DET', 'ADJ', 'NOUN', 'CCONJ', 'VERB', 'PART', 'ADP', 'ADV', 'PROPN', 'PRON', 'NUM', 'INTJ', 'X', 'SYM')
        with open(testset,encoding='utf-8', mode = 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            line = ''
            for row in reader:
                if line != row[1]:
                    line = row[1]
                    sentence = []
                    lemma_sentence = []
                    pos_list = []
                    for word in nlp(row[1]):
                        if word.is_alpha:
                            sentence.append(word.text)
                            lemma_sentence.append(word.lemma_)
                            pos_list.append(word.pos_)
                
                label = row[9]
                ori_words = row[4]
                target_words = nlp(ori_words)

                feature = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]             
                if len(target_words) == 1:
                    word = ori_words
                    try:
                        index = sentence.index(word)
                    except:
                        if word=='madness':
                            index = lemma_sentence.index(word)
                        elif word=='sv':
                            index = lemma_sentence.index(word)
                        else:
                            continue
                    pos = pos_list[index]
                    pos_index = pos_set.index(pos)
                    length,freq,syllable,synonyms = self.model.extract_features([lemma_sentence[index]])
                    
                    feature[pos_index] = 1
                    feature.append(length)
                    feature.append(freq)
                    feature.append(syllable)
                    feature.append(synonyms)
                    #if label != self.clf.predict([feature]):
                    #    print(word,label,feature[-5:])                       
                    self.gold_labels.append(label)
                    self.predicted_labels.append(self.clf.predict([feature]))
                else:
                    words = []
                    for word in target_words:
                        if word.is_alpha:
                            try:
                                index = sentence.index(word.text)
                            except:
                                if word.text =='madness':
                                    index = lemma_sentence.index(word.text)
                                elif word.text =='sv':
                                    index = lemma_sentence.index(word.text)
                                else:
                                    continue
                            pos = pos_list[index]
                            pos_index = pos_set.index(pos)
                            feature[pos_index] = 1
                            words.append(lemma_sentence[index])
                    length,freq,syllable,synonyms = self.model.extract_features(words)
                    feature.append(length)
                    feature.append(freq)
                    feature.append(syllable)
                    feature.append(synonyms)
                    #if label != self.clf.predict([feature]):
                        #print(words,label,feature[-5:])                       
                    self.gold_labels.append(label)
                    self.predicted_labels.append(self.clf.predict([feature]))

    def report_score(self,detailed=True):
        macro_F1 = sklearn.metrics.f1_score(self.gold_labels, self.predicted_labels, average='macro')
        print("macro-F1: {:.2f}".format(macro_F1))
        if detailed:
            scores = sklearn.metrics.precision_recall_fscore_support(self.gold_labels, self.predicted_labels)
            print("{:^10}{:^10}{:^10}{:^10}{:^10}".format("Label", "Precision", "Recall", "F1", "Support"))
            print('-' * 50)
            print("{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(0, scores[0][0], scores[1][0], scores[2][0], scores[3][0]))
            print("{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(1, scores[0][1], scores[1][1], scores[2][1], scores[3][1]))
        print()
    
    

class Spa_train():
    def __init__(self,model,trainset):
        self.clf = svm.SVC(kernel='rbf')
        self.model = model
        self.features = None
        self.labels = None
        self.set_fea_labels(trainset)
        
        self.predicted_labels = []
        self.gold_labels = []
        
    def set_fea_labels(self,trainset):
        pos_set = ('PUNCT', 'DET', 'AUX', 'ADV', 'ADJ', 'PART', 'PROPN', 'PRON', 'NUM', 'NOUN', 'VERB', 'SYM', 'SCONJ', 'ADP', 'CONJ')
        features = []
        labels = []
        with open(trainset,encoding='utf-8', mode = 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            line = ''
            for row in reader:
                if line != row[1]:
                    line = row[1]
                    sentence = []
                    lemma_sentence = []
                    pos_list = []
                    for word in nlp(row[1]):
                        if word.is_alpha:
                            sentence.append(word.text)
                            lemma_sentence.append(word.lemma_)
                            pos_list.append(word.pos_)
                
                label = row[9]
                ori_words = row[4]
                target_words = nlp(ori_words)

                feature = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]             
                if len(target_words) == 1:
                    word = ori_words
                    try:
                        index = sentence.index(word)
                    except:
                        continue
                    
                    pos = pos_list[index]
                    pos_index = pos_set.index(pos)
                    length,freq,syllable,synonyms = self.model.extract_features([lemma_sentence[index]]) #
                    
                    feature[pos_index] = 1
                    feature.append(length)
                    feature.append(freq)
                    feature.append(syllable)
                    feature.append(synonyms)
                    
                    features.append(feature)
                    labels.append(label)
                    
                else:
                    words = []
                    for word in target_words:
                        if word.is_alpha:
                            try:
                                index = sentence.index(word.text)
                            except:
                                continue
                            pos = pos_list[index]
                            pos_index = pos_set.index(pos)
                            feature[pos_index] = 1
                            words.append(lemma_sentence[index])
                    length,freq,syllable,synonyms = self.model.extract_features(words) #
                    feature.append(length)
                    feature.append(freq)
                    feature.append(syllable)
                    feature.append(synonyms)
                    
                    features.append(feature)
                    labels.append(label)
            self.features = features
            self.labels = labels
            
    def train(self):
        #cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        #title = 'Learning Curves'
        #estimator = svm.SVC(kernel='rbf')
        #self.plot_learning_curve(estimator, title, self.features, self.labels,(0.7, 1.01), cv=cv, n_jobs=4)
        #plt.show()
        self.clf.fit(self.features, self.labels) 
        
    def dev(self,testset):
        pos_set = ('PUNCT', 'DET', 'AUX', 'ADV', 'ADJ', 'PART', 'PROPN', 'PRON', 'NUM', 'NOUN', 'VERB', 'SYM', 'SCONJ', 'ADP', 'CONJ','INTJ')
        with open(testset,encoding='utf-8', mode = 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            line = ''
            for row in reader:
                if line != row[1]:
                    line = row[1]
                    sentence = []
                    lemma_sentence = []
                    pos_list = []
                    for word in nlp(row[1]):
                        if word.is_alpha:
                            sentence.append(word.text)
                            lemma_sentence.append(word.lemma_)
                            pos_list.append(word.pos_)
                
                label = row[9]
                ori_words = row[4]
                target_words = nlp(ori_words)

                feature = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]             
                if len(target_words) == 1:
                    word = ori_words
                    try:
                        index = sentence.index(word)
                    except:
                        continue

                    pos = pos_list[index]
                    pos_index = pos_set.index(pos)
                    length,freq,syllable,synonyms = self.model.extract_features([lemma_sentence[index]]) #
                    feature[pos_index] = 1
                    feature.append(length)
                    feature.append(freq)
                    feature.append(syllable)
                    feature.append(synonyms)

                    #if label != self.clf.predict([feature]):
                    #    print(ori_words)
                    
                    self.gold_labels.append(label)
                    self.predicted_labels.append(self.clf.predict([feature]))
                else:
                    words = []
                    for word in target_words:
                        check = False
                        if word.is_alpha:
                            try:
                                index = sentence.index(word.text)
                            except:
                                check = True
                                break
                            pos = pos_list[index]

                            pos_index = pos_set.index(pos)

                            feature[pos_index] = 1
                            words.append(lemma_sentence[index])
                    if check:
                        continue                            
                    length,freq,syllable,synonyms = self.model.extract_features(words) #
                    feature.append(length)
                    feature.append(freq)
                    feature.append(syllable)
                    feature.append(synonyms)
                    
                    #if label != self.clf.predict([feature]):
                     #   print(words)
                    
                    self.gold_labels.append(label)
                    self.predicted_labels.append(self.clf.predict([feature]))

    def plot_learning_curve(self,estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    
        plt.legend(loc="best")
        return plt  
    
    def report_score(self,detailed=True):
        macro_F1 = sklearn.metrics.f1_score(self.gold_labels, self.predicted_labels, average='macro')
        print("macro-F1: {:.2f}".format(macro_F1))
        if detailed:
            scores = sklearn.metrics.precision_recall_fscore_support(self.gold_labels, self.predicted_labels)
            print("{:^10}{:^10}{:^10}{:^10}{:^10}".format("Label", "Precision", "Recall", "F1", "Support"))
            print('-' * 50)
            print("{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(0, scores[0][0], scores[1][0], scores[2][0], scores[3][0]))
            print("{:^10}{:^10.2f}{:^10.2f}{:^10.2f}{:^10}".format(1, scores[0][1], scores[1][1], scores[2][1], scores[3][1]))
        print()
        
if __name__ == '__main__':
    
    language = 'spanish'
    train_path = 'Spanish_Train.tsv'
    test_path = 'Spanish_Dev.tsv'
    
    if language == 'english':
        nlp = spacy.load('en')
        special_case = [{ORTH: u'madness"((sv', LEMMA: u'madness', POS:'NOUN',IS_ALPHA:True},{ORTH: u'"'},{ORTH: u'('},{ORTH: u'('},{ORTH: u'sv'}]
        nlp.tokenizer.add_special_case(u'madness"((sv', special_case)
        special_case = [{ORTH: u'life"((sv',  LEMMA: u'life', POS:'NOUN',IS_ALPHA:True},{ORTH: u'"'},{ORTH: u'('},{ORTH: u'('},{ORTH: u'sv'}]
        nlp.tokenizer.add_special_case(u'life"((sv', special_case)
        special_case = [{ORTH: u"n't", LEMMA: u"not", POS:'ADV',IS_ALPHA:True}]
        nlp.tokenizer.add_special_case(u"n't", special_case)  
        special_case = [{ORTH: u"’re", LEMMA: u"are", POS:'VERB',IS_ALPHA:True}]
        nlp.tokenizer.add_special_case(u"’re", special_case)
        special_case = [{ORTH: u"'s", LEMMA: u"is", POS:'VERB',IS_ALPHA:True}]
        nlp.tokenizer.add_special_case(u"'s", special_case)
        
        start = time.time()
        Eng_features = features(train_path,language)
        test = Eng_train(Eng_features,train_path)
        test.train()
        test.dev(test_path)
        test.report_score()
        print(int(time.time()-start),'s')    
        
    elif language == 'spanish':
        nlp = spacy.load('es_core_news_sm')
        special_case = [{ORTH: u"d'Ullà", LEMMA: u"d'Ullà", POS:'NOUN',IS_ALPHA:True}]
        nlp.tokenizer.add_special_case(u"d'Ullà", special_case)
        
        start = time.time()
        Spa_features = features(train_path,language)
        test = Spa_train(Spa_features,train_path)
        test.train()
        test.dev(test_path)
        test.report_score()
        print(int(time.time()-start),'s')
        
    else:
        print('English or Spanish Only')        
