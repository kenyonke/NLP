import math
import sys
class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index,termWeighting):
        self.index = index
        self.termWeighting = termWeighting
    # Method to apply query to index
    def forQuery(self,query):
        if self.termWeighting=='binary':
            rank={}
            Q={}
            D={}
            # Q={term1:binary1,term2:binary2,...}
            for term in query:
                Q[term]=1
            # D={docid1:{d1term1:d1binary1,d1term2:d1binary2,...},docid2...}
            for term in self.index:
                for docid in self.index[term]:
                    D[docid]={}
            for term in self.index:
                for docid in self.index[term]:
                    D[docid][term]=1
            # vector model
            for docid in D:
                cos = 0
                d = 0
                for term,binary in D[docid].items():
                    d+=binary*binary
                d=math.sqrt(d)
                for term,binary in Q.items():
                    if term not in D[docid]:
                        D[docid][term]=0
                    cos+=binary*D[docid][term]
                rank[docid]=cos/d
            rank=sorted(rank.items(),key = lambda i:i[1],reverse = True)
            result = list(dict(rank))
        
        if self.termWeighting=='tf':
            rank={}
            D={}
            Q={}
            # query={term1:freq1,term2:freq2,...}
            # D={docid1:{d1term1:d1freq1,d1term2:d1freq2,...},docid2...}
            for term in self.index:
                for docid in self.index[term]:
                    D[docid]={}
            for term in self.index:
                for docid,freq in self.index[term].items():
                    D[docid][term]=freq
            #vector model
            for docid in D:
                cos = 0
                d = 0
                for term,freq in D[docid].items():
                    d+=freq*freq
                d=math.sqrt(d)
                for term,freq in query.items():
                    if term not in D[docid]:
                        D[docid][term]=0
                    cos+=freq*D[docid][term]
                rank[docid]=cos/d
            rank=sorted(rank.items(),key = lambda i:i[1],reverse = True)
            
        if self.termWeighting=='tfidf':
            rank={}
            Q={}
            D={}
            TI={}   # dict TI is a collection of tfidf of all the documents in index
            DD=3204 # DD is total number of all the documents in collection 
            # Q = {term1:tfidf1,term2:tfidf2,...}
            for term,freq in query.items():
                df=0
                if term not in self.index:
                    Q[term]=0
                else:
                    for docid in self.index[term]:
                        df+=1
                    Q[term]=freq*(math.log(DD/df))
            # part 1 D={docid1:{d1term1:d1freq1,d1term2:d1freq2,...},docid2...}
            for term in self.index:
                for docid in self.index[term]:
                    D[docid]={}
                    TI[docid]={}
            for term in self.index:
                for docid,freq in self.index[term].items():
                    D[docid][term]=freq
                    
            # part 2 TI={docid1:{d1term1:d1tfidf1,d1term2:d1tfidf2,...},docid2...}
            for docid in D:
                for term,freq in D[docid].items():
                    df=0  # df is number of documents containing a term
                    for key in self.index[term].keys():
                        df+=1
                    TI[docid][term]=freq*math.log(DD/df)
                    
            #vector space model
            for docid in TI:
                cos = 0
                d = 0 # d is di in the formula
                # calculate |d|
                for term,tfidf in TI[docid].items():
                    d+=tfidf*tfidf
                d=math.sqrt(d)
                # calculate similarity feature cos
                for term,tfidf in Q.items():
                    #add a term in dict TI if the term is not in dict TI
                    if term not in TI[docid]:
                        TI[docid][term]=0
                    #calculation the sigma of the tdidf formula
                    cos+=tfidf*TI[docid][term]
                rank[docid]=cos/d
            rank=sorted(rank.items(),key = lambda i:i[1],reverse = True)
        #print '#' after one query retrieval
        sys.stdout.write("#")
        sys.stdout.flush()
        #transform rank to list only with its keys
        result = list(dict(rank))
        return result
