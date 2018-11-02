"""\
------------------------------------------------------------
USE: python <PROGNAME> (options) 
OPTIONS:
    -h : print this help message
    -s : compression method choice (char or word, default: char)
    -I PATT : identify input files
------------------------------------------------------------
"""


class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:],'hs:I:')
        opts = dict(opts)

##############################
# HELP option

        if '-h' in opts:
            help = __doc__.replace('<PROGNAME>',sys.argv[0],1)
            print(help,file=sys.stderr)
            sys.exit()
##############################
# Identify input files, when "-I" option used

        if '-I' in opts:
            self.filename = opts['-I']
        else:
            print(' ERROR: No input file, please see the help information ',file=sys.stderr)
            sys.exit()
##############################
# syntax option

        if '-s' in opts:
            if opts['-s'] in ('char','word'):
                self.syntax = opts['-s']
            else:
                warning = (
                        "*** ERROR: synatx label (opt: -s syntx) not recognised! ***\n"
                        " -- must be one of: char / word, please see the help information ")
                print(warning, file=sys.stderr)
                sys.exit()
##############################


# -*- coding: utf-8 -*-

#prob function calculate the probabilities of all the terms(depend on syntax choice)
#finally, output these data as a dict.
def prob(syntax,filename):
    if syntax == 'char':
        pattern = re.compile('[\w\W]')
    if syntax == 'word':
        pattern = re.compile('[\w]+|[\W]')
    probStore = {}
    with open(filename,'r') as f:
        s = 0 
        for line in f:
            result = pattern.findall(line)
            for term in result:
                if term not in probStore.keys():
                    probStore[term] = 1
                else:
                    probStore[term] +=1
                s+=1
        #add pseudo symbol into probabilities dict 
        probStore['■'] = 1
        s += 1
        for term,freq in probStore.items():
            probStore[term] = freq/s
        return probStore
    
#creat a Node class which is used for creating huffman tree 
class Node:
    def __init__(self,symbol,probability):
        self.symbol = symbol
        self.prob  = probability
        self.leftNode = None
        self.rightNode = None
        self.fatherNode = None 

# a function which can output a list composed of Node type data if inputing a probabilities dict
def node(probDict):
    nodeList = []
    for term,prob in probDict.items():
        nodeList.append(Node(term,prob))    
    return nodeList

# tree function can creat huffman tree and return the top point if inputting a node list
def tree(nodeList):
    nodeList.sort(key = lambda node:node.prob)
    nodeListTrain = nodeList[:]
    while(len(nodeListTrain) > 1):
        leftNode = nodeListTrain.pop(0)
        rightNode = nodeListTrain.pop(0)
        #creat a new node which is called fatherNode for the two lowest probability nodes
        fatherNode = Node(None,leftNode.prob+rightNode.prob)  
        fatherNode.leftNode = leftNode
        fatherNode.rightNode = rightNode
        leftNode.fatherNode = fatherNode
        rightNode.fatherNode = fatherNode

        #rather than sorting the list again, inserting the new fatherNode into list is better
        if len(nodeListTrain) != 0:
            # when the length of trainning list becomes 1
            if len(nodeListTrain) == 1:
                if fatherNode.prob< nodeListTrain[0].prob:
                    nodeListTrain.insert(0,fatherNode)
                else:
                    nodeListTrain.insert(1,fatherNode)
            
            # when the length of trainning list is longer than 2, using dichotomy algorithm to improve efficiency   
            else:
                #initialise the start index and end index from node list
                start = 0
                end = len(nodeListTrain)-1
                index = 0           #index is the location we want to insert into node list
                while(start <= end):
                    mindex = int((end+start)/2)
                    mprob = nodeListTrain[mindex].prob
                    #if only two node left, comparing them with father node we create in order to find its index
                    if end-start ==1:
                        if fatherNode.prob >= nodeListTrain[end].prob:
                            index = end + 1
                        elif fatherNode.prob <= nodeListTrain[start].prob:
                            index = start
                        else:
                            index = end
                        break
                    #compare middle probability of list with father node we create and choose which side the father node belongs.
                    if fatherNode.prob == mprob:
                        index = mindex
                        break
                    elif fatherNode.prob > mprob:
                        start = mindex 
                    elif fatherNode.prob < mprob:
                        end = mindex
                nodeListTrain.insert(index,fatherNode)
        else:
            nodeListTrain.append(fatherNode)
    return nodeListTrain[0]

# binaryCode function create the binary code(eg.'010101') for each term.
def binaryCode(nodeList, topNode):
    #initialise dict code whose keys are symbol and values are corresponding binary numbers.
    code = {}
    import pickle
    # design binary number for each symbol
    for node in nodeList:
        code[node.symbol] = ''
    for node in nodeList:
        loopNode = node
        while(loopNode != topNode):
            if loopNode.fatherNode.leftNode == loopNode:
                code[node.symbol] = '0' + code[node.symbol]
            else:
                code[node.symbol] = '1' + code[node.symbol]
            loopNode = loopNode.fatherNode
    # store compress model into a file whose name is 'infile_symbol_model.pkl'
    with open('infile_symbol_model.pkl','wb') as f:
        pickle.dump(code,f)
    return code

def compress(code,filename,pattern):
    import array
    #initialise dict code which is used for sequentially storing binary numbers of all the symbols in the input text 
    t = ''
    codearray = array.array('B')
    
    with open(filename,'r') as file:
        #char compress
        if pattern == 'char':
            for line in file:
                for char in line:
                    t += code[char]
            t += code['■']
            if (len(t)%8) != 0:
                t += '0' * (8-len(t)%8)
            for i in range(int(len(t)/8)):
                codearray.append(int(t[i*8:i*8+8],2))
            with open('workfile','wb') as f:
                codearray.tofile(f)         
 
        # word compress
        if pattern == 'word':
            pattern = re.compile('[\w]+|[\W]')
            words = file.read()
            words =pattern.findall(words)
            for word in words:
                t += code[word]
            t += code['■']
            if (len(t)%8) != 0:
                t += '0' * (8-len(t)%8)
            for i in range(int(len(t)/8)):
                codearray.append(int(t[i*8:i*8+8],2))
            with open('workfile','wb') as f:
                codearray.tofile(f)  

if __name__ == '__main__':
    import sys, re, getopt, time
    config = CommandLine()
    model_s = time.time()
    probStore = prob(config.syntax,config.filename)
    nodeList = node(probStore)
    topNode = tree(nodeList)
    code = binaryCode(nodeList,topNode)
    model_e = time.time()
    compress(code,config.filename,config.syntax)
    end = time.time()
    print('Time spending of building symbol model is ',model_e-model_s,'s')
    print('Time spending of encoding given the symbol model is ',end - model_e,'s')
