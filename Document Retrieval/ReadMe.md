The goal of this work is finding the relevant documents based on the queries.

We make use of binary, tf and tfidf mode to achieve that.

The materials provided include a file 'documents.txt', which contains a collection of documents that record publications in the CACM.
Every row in preprocessed documents ('index_nostoplist_nostemming.txt', 'index_nostoplist_withstemming.txt', 'index_withstoplist_nostemming.txt', 'index_withstoplist_withstemming.txt') is represented as: **specific_word document_id1:counts document_id2:counts ...**

The file 'queries.txt' contains a set of IR queries for use against this collection.
Every row in preprocessed queries ('queries_nostoplist_nostemming.txt', 'queries_nostoplist_withstemming.txt', 'queries_withstoplist_nostemming.txt', 'queries_withstoplist_withstemming.txt') is represented as: **query_index word1:counts word2:counts ...**

The file cacm_gold_std.txt is a 'gold standard' identifying the documents that have been judged relevant to each query.

There are also some different processing methods based queryies and documents included such as stopwords and steaming and they can be choosed in the code.

## Runing code example
```
python ir_engine.py -o outputfile -w binary
```
more information 
```
python ir_engine.py -h
```

## evaluation
```
python eval_ir.py outputfile cacm_gold_std.txt
```

## Experiment Results
<img src="https://github.com/kenyonke/NLP/blob/master/Document%20Retrieval/results.png" width="500" height="400">

The performance scores by comparing system result files to collection gold standard document are shown above. From the table we can see that preprocess (stop list and stemming) can improve performance of retrieval. Furthermore, tfidf mode is more accurate than other two modes and tf mode is better than binary mode. However, the speed of binary mode is faster and tfidf is the slowest and most complex mode in practice. In addition, performances of all the three modes is not good, the reason could be that single mode is not enough for retrieval.
