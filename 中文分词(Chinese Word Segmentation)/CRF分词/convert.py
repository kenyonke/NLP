# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 01:09:55 2018
@author: kenyon
"""
import codecs

def writeWordTag(word,length,file):
    if(length==1):
        file.write(word+' '+ 'S\n' )
    else:
        for i in range(length):
            if(i==0):
                file.write(word[i]+' '+'B\n')
            elif(i!=length-1):
                file.write(word[i]+' '+'M\n')
            else:
                file.write(word[i]+' '+'E\n')

def countLines(file):
    count = -1
    for count, line in enumerate(codecs.open(file, 'rU', 'utf-8')):
        pass
    return count+1

def convert_PeopleDaily_To_Train(file,output1,output2):
    cols = countLines(file)
    trainCols = int(cols*0.7)
    f = codecs.open(file,'r',encoding='UTF-8')
    trainFile = codecs.open(output1, 'w',encoding='UTF-8')
    testFile =  codecs.open(output2, 'w',encoding='UTF-8')
    
    #write training file
    for index,line in enumerate(f):
        wordTag = line.split()[1:]
        for wt in wordTag:
            word,tag = wt.split('/')
            writeWordTag(word,len(word),trainFile)
        if index== trainCols:
            break
        trainFile.write('\n')   
    trainFile.close()
    
    #write testing file
    for line in f:
        wordTag = line.split()[1:]
        for wt in wordTag:
            word,tag = wt.split('/')
            writeWordTag(word,len(word),testFile)
        testFile.write('\n')    
    testFile.close()
    f.close()    

if __name__ == '__main__':
    convert_PeopleDaily_To_Train('people-daily.txt','CHN-train','CHN-test')  