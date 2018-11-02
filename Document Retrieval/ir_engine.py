"""\
------------------------------------------------------------
USE: python <PROGNAME> (options)
OPTIONS:
    -h : print this help message
    -s : use "with stoplist" configuration (default: without)
    -p : use "with stemming" configuration (default: without)
    -w LABEL : use weighting scheme "LABEL" (LABEL in {binary, tf, tfidf}, default: binary)
    -o FILE : output results to file FILE (default: output to stdout)
------------------------------------------------------------\
"""

import sys, getopt, re
from my_retriever import Retrieve

class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:],'hspw:o:')
        opts = dict(opts)

        if '-h' in opts:
            self.printHelp()

        if len(args) > 0: #不需要输入files，这个类只定义了一些设置
            print("*** ERROR: no arg files - only options! ***", file=sys.stderr)
            self.printHelp()

        if '-w' in opts:
            if opts['-w'] in ('binary','tf','tfidf'):
                self.termWeighting = opts['-w']     #定义其中的termWeighting方式
            else:
                warning = (
                    "*** ERROR: term weighting label (opt: -w LABEL) not recognised! ***\n"
                    " -- must be one of: binary / tf / tfidf")
                print(warning, file=sys.stderr)
                self.printHelp()
        else:
            self.termWeighting = 'binary' #如果没有值，默认为binary

        if '-o' in opts:
            self.outfile = opts['-o']   #定义outfile
        else:
            self.outfile = None

        if '-s' in opts and '-p' in opts:   #选择4种不同的model的其中一种
            self.indexFile = 'documents/index_withstoplist_withstemming.txt'
            self.queriesFile = 'queries/queries_withstoplist_withstemming.txt'
        elif '-s' in opts:
            self.indexFile = 'documents/index_withstoplist_nostemming.txt'
            self.queriesFile = 'queries/queries_withstoplist_nostemming.txt'
        elif '-p' in opts:
            self.indexFile = 'documents/index_nostoplist_withstemming.txt'
            self.queriesFile = 'queries/queries_nostoplist_withstemming.txt'
        else:
            self.indexFile = 'documents/index_nostoplist_nostemming.txt'
            self.queriesFile = 'queries/queries_nostoplist_nostemming.txt'

    def printHelp(self):   #output the help information 
        help = __doc__.replace('<PROGNAME>',sys.argv[0],1)
        print(help, file=sys.stderr)
        sys.exit()

class Queries:                        
    def __init__(self,queriesFile):          #整个目的就是将queries整合到一个二维dict--(self.qStore[qid][term])里面去。格式为{1:{qid1},2:{qid2},3:{qid3}...},其中每个qid={term:count,...}
        self.qStore = {}                        #定义 qStore
        termCountRE = re.compile('(\w+):(\d+)') #定义一个检测pattern，见笔记lab2, '(\w+):(\d+)'？？？？
        f = open(queriesFile,'r')
        for line in f:
            qid = int(line.split(' ',1)[0])  #这里定义了qid
                                             #str.split(str="",num=string.count(str))[n]
                                             #num：表示分割次数。如果存在参数num,则仅分隔成num+1个子字符串，并且每一个子字符串可以赋给新的变量
                                             #str:表示为分隔符，默认为空格，但是不能为空('')。若字符串中没有分隔符，则把整个字符串作为列表的一个元素
                                             #[n]：表示选取第n个分片
            #according to 'queries_nostoplist_nostemming.txt', 这里的qid为每line前面的index，'(\w+):(\d+)'应该为format-word:count，跟txt里一样的格式
            self.qStore[qid] = {}
            for (term,count) in termCountRE.findall(line): #pattern的findall见lab
                self.qStore[qid][term] = int(count)
    
    def getQuery(self,qid):              #获取选定index的dict信息
        if qid in self.qStore:
            return self.qStore[qid]
        else:
            print("*** ERROR: unknown query identifier (\"%s\") ***" % qid, file=sys.stderr)
            if type(qid) == type(''): #正确操作的示范，废话
                print('WARNING: query identifiers should be of type: integer', file=sys.stderr)
                print('         -- your query identifier is of type: string', file=sys.stderr)
            print(' -- program exiting', file=sys.stderr)
    
    def qids(self):                         #定义qids() 
        return sorted(self.qStore.keys())   #dict.keys() 用list返回一个字典所有key ie.['key1','key2','key3',...]

class IndexLoader:
    def __init__(self,indexFile):               #将index文件整合到一个二维dict里面去。
        self.index = {}
        docidCountRE = re.compile('(\d+):(\d+)')#  pattern
        f = open(indexFile,'r')
        for line in f:
            term = line.split(' ',1)[0]         #分割字符，取第一个，index文件里应该为第一列的单词
            self.index[term] = {}               #create new dict, ie. index={a:{},aaron:{},abac:{}......}
            for (docid,count) in docidCountRE.findall(line):
                docid = int(docid)
                self.index[term][docid] = int(count)

    def getIndex(self):                 #输出index二维dict
        return self.index

class ResultStore:                      #保存result files
    def __init__(self,outfile):
        self.outfile = outfile
        self.results = []

    def store(self,qid,docids):         #取前十，加到result里面去.
        if len(docids) > 10:
            docids = docids[:10]
        self.results.append((qid,docids))

    def output(self):
        if self.outfile:
            with open(self.outfile,'w') as out:
                self.output_main(out)
        else:
            self.output_main(sys.stdout)

    def output_main(self,outstream):
        for (qid,docids) in self.results:
            for docid in docids:
                print(qid, docid, file=outstream)

if __name__ == '__main__':
    import time
    s = time.time()
    config = CommandLine()
    indexLoader = IndexLoader(config.indexFile)
    index = indexLoader.getIndex()
    retrieve = Retrieve(index,config.termWeighting)
    queries = Queries(config.queriesFile)
    allResults = ResultStore(config.outfile)

    for qid in queries.qids():
        query = queries.getQuery(qid)
        results = retrieve.forQuery(query)
        allResults.store(qid,results)
        
    allResults.output()
    e = time.time()
    print()
    print('runing time: ',e-s)
