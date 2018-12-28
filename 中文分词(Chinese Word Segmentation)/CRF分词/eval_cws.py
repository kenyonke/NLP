# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:38:10 2018

@author: kenyon
"""
import codecs

def evaluation(file):
    file = codecs.open('result', "r",encoding='UTF-8')  
    wc_of_test = 0  
    wc_of_gold = 0  
    wc_of_correct = 0  
    flag = True  
      
    for l in file:  
        if len(l)==2:
            continue
        #print(l,end='-')
        #print(len(l))
        #print(l.strip().split())
        _, g, r = l.strip().split()  
       
        if r != g:  
            flag = False  
      
        if r in ('E', 'S'):  
            wc_of_test += 1  
            if flag:  
                wc_of_correct +=1  
            flag = True  
      
        if g in ('E', 'S'):  
            wc_of_gold += 1  
  
    print("WordCount from test result:", wc_of_test)
    print("WordCount from golden data:", wc_of_gold)
    print("WordCount of correct segs :", wc_of_correct)
              
    #precision  
    P = wc_of_correct/float(wc_of_test)  
    #recall  
    R = wc_of_correct/float(wc_of_gold)  
      
    print("P = %f, R = %f, F-score = %f" % (P, R, (2*P*R)/(P+R)))

if __name__ == '__main__':
    evaluation('result')
