# -*- coding: utf-8 -*-
import codecs
import nltk
from collections import Counter
import re
import sys

class CommandLine:
    def __init__(self):
        if (sys.argv[1] == '' or sys.argv[2] == ''):
            print('please type file path')
            sys.exit()
        else:
            self.corpus_filepath = sys.argv[1]
            self.questions_filepath = sys.argv[2]
        
class model:
    def __init__(self,filepath):
        data = self.load_file(filepath)
        #two models of unigram and bigram which are dicts storing frequencies of words
        self.unigram_counts = data[0]
        self.bigram_counts = data[1]
        #V is the total numbers of sentences 
        self.V = data[2]
        #num is the total numbers of words
        self.num = data[3]
    
    #input filepath and return unigram's model and bigram's model
    def load_file(self,filepath):
        bigrams = []
        unigram_counts = {}
        num = 0
        V = 0
        f = codecs.open(filepath,'r','utf-8')
        pattern = re.compile('[\w]+|[W]')
        for line in f:
            V += 1
            #convert words to lower case
            new_line = pattern.findall(line.lower())
            bigrams.extend(nltk.bigrams(new_line, pad_left=True, pad_right=True))
            for word in new_line:
                num += 1
                if word in unigram_counts.keys():
                    unigram_counts[word] += 1
                else:
                    unigram_counts[word] = 1
        bigram_counts = Counter(bigrams)
        f.close()
        return unigram_counts,bigram_counts,V,num
    
    #smoothing method
    def bigram_LM(self,sentence_x, smoothing=0.0):
        unique_words = len(self.unigram_counts.keys()) + 2 # For the None paddings
        x_bigrams = nltk.bigrams(sentence_x, pad_left=True, pad_right=True)
        prob_x = 1.0
        for bg in x_bigrams:
            if bg[0] == None:
                prob_bg = (self.bigram_counts[bg]+smoothing)/(self.V+smoothing*unique_words)
            else:
                prob_bg = (self.bigram_counts[bg]+smoothing)/(self.unigram_counts[bg[0]]+smoothing*unique_words)
            prob_x = prob_x *prob_bg
            #print(str(bg)+":"+str(prob_bg))
        return prob_x
    
    def unigram(self,sentence_x):
        prob_x = 1.0
        for un in sentence_x:
            prob_x = prob_x * self.unigram_counts[un] / self.num
        return prob_x

def compare(prob1,prob2,question_line,answer_words,_index,model_type):
    if prob1>prob2:
        question_line[_index] = "'"+answer_words[0]+"'"
    else:
        question_line[_index] = "'"+answer_words[1]+"'"
    sentence = ''
    for word in question_line:
        sentence = sentence + word + ' '
    print(model_type,':',sentence)
    print(prob1,'  ',prob2)

#corpus_file,question_file
def operate(question_filepath,model):
    pattern = re.compile('[\w]+|[W]')
    questions = []
    answers = []
    answer_index = []
    f = open(question_filepath,'r')
    for line in f:
        new_line = pattern.findall(line.lower())
        #_index means index of '____'
        _index = 0
        for word in new_line:
            if new_line[_index] == '____':
                break
            else:
                _index += 1
        question_line = new_line[0:-2]
        answer_words = [new_line[-1],new_line[-2]]
        questions.append(question_line)
        answers.append(answer_words)
        answer_index.append(_index)
    f.close()
    
    for i in range(len(questions)):
        a_sents = []
        b_sents = []
        #if the answer is on the middle of the question
        if answer_index[i] != 0 and answer_index[i] != (len(questions[i])-1):
            a_sents = questions[i][:answer_index[i]]
            a_sents.append(answers[i][0])
            a_sents += questions[i][answer_index[i]+1:]
    
            b_sents = questions[i][:answer_index[i]]
            b_sents.append(answers[i][1])
            b_sents += questions[i][answer_index[i]+1:]
        #if the answer is on the beginning of the question
        elif answer_index[i] == 0:
            a_sents.append(answers[i][0])
            a_sents += questions[i][1:]
            
            b_sents.append(answers[i][1])
            b_sents += questions[i][1:]
        #if the answer is on the end of the question
        else:
            a_sents += questions[i][:-1]
            a_sents.append(answers[i][0])
            
            b_sents += questions[i][:-1]
            b_sents.append(answers[i][1])
        
        a_sentence = ''
        for word in a_sents:
            a_sentence = a_sentence + word + ' '
            
        b_sentence = ''
        for word in b_sents:
            b_sentence = b_sentence + word + ' ' 
        
        print(answers[i])
        #unigram model
        a = model.unigram(a_sents)
        b = model.unigram(b_sents)
        compare(a,b,questions[i],answers[i],answer_index[i],'UI')
        
        #bigram model
        a = model.bigram_LM(a_sents, smoothing=0.0)
        b = model.bigram_LM(b_sents, smoothing=0.0)
        compare(a,b,questions[i],answers[i],answer_index[i],'BG')
        
        #'bigram model with smoothing
        a = model.bigram_LM(a_sents, smoothing=1.0)
        b = model.bigram_LM(b_sents, smoothing=1.0)
        compare(a,b,questions[i],answers[i],answer_index[i],'BS')
        
        print('')
if __name__ == '__main__':
    config = CommandLine()
    model = model(config.corpus_filepath)
    operate(config.questions_filepath,model)

