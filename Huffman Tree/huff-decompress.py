"""\
------------------------------------------------------------
USE: python <PROGNAME> (options) 
OPTIONS:
    -h : print this help message
    -I PATT : identify compressed files
    -M PATT : identify model
------------------------------------------------------------
"""


class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:],'hM:I:')
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
# Identify input files, when "-I" option used

        if '-M' in opts:
            self.model = opts['-M']
        else:
            print(' ERROR: No model file, please see the help information ',file=sys.stderr)
            sys.exit()
##############################


# -*- coding: utf-8 -*-
def modelChange(model):
    sort = {}
    #create new model whose form is {length:{binary number:symbol,...},...}
    for term,code in model.items():
        length = len(code)
        if length not in sort.keys():
            sort[length] = {}
            sort[length][code] = term
        else:
            sort[length][code] = term

    #initialise the shortest length of code and the longest length of code
    lonLen = 0
    for term,code in model.items():
        shoLen = len(code)
        break
    # calculate the shortest length of code and the longest length of code
    for term,code in model.items():
        if (len(code)) > lonLen:
            lonLen = len(code)
        if len(code) < shoLen:
            shoLen = len(code)
    #add lacks of length between the shortest length and longest length into newmodel's key
    #to make sure that all the lengths in the keys of new model 
    for length in range(shoLen,lonLen+1):
        if length not in sort.keys():
            sort[length] = {}
    return sort,shoLen,lonLen

def decompress(filename,model,shoLen,lonLen):
    #1)load compressed file and put all binary numbers in 'file'(str type).
    file = ''
    f = open(filename,'rb')
    i=0
    for char in f.read():
        char = str(bin(char).replace('0b',''))
        #add 0 in front of binary numbers whihch is less than 8 
        if len(char)< 8:
            char = '0' * ( 8 - len(char)) +char
        file += char
    #2)use 'file' to start decompression with changed model and put decompressed words into 'defile'(str type)
    defile = ''
    loc = 0
    #start decompress
    while True:
        for i in range(shoLen,lonLen+1):
            if file[loc:loc+i] in model[i].keys():
                word = model[i][file[loc:loc+i]]
                loc += i
                break
        #stop decompression when encountering '■'
        if word == '■':
            break
        defile = defile + word
    #3)witre 'defile' into document 'decompress_huffman_file.txt'
    f = open('decompress_huffman_file.txt','w')
    f.write(defile)
    f.close()

if __name__ == '__main__':
    import sys, getopt, pickle, time
    start = time.time()
    config = CommandLine()
    with open(config.model,'rb') as loadModel:
        model = pickle.load(loadModel)
    loadModel.close()
    newModel,shoLen,lonLen = modelChange(model)
    c = decompress(config.filename,newModel,shoLen,lonLen)
    end = time.time()
    print ('Time spending of decoding is ',end - start ,'s')
